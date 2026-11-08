from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASETS = {
    "edge": {
        "path": Path(os.environ.get("DP_SPSR_EDGE_CSV", "data/Edge-IIoTset/ML-EdgeIIoT-dataset.csv")),
        "label": "Attack_label",
        "aux": "Attack_type",
        "numeric": [
            "arp.opcode", "arp.hw.size", "icmp.seq_le", "http.content_length",
            "http.response", "tcp.connection.fin", "tcp.connection.rst",
            "tcp.connection.syn", "tcp.connection.synack", "tcp.dstport",
            "tcp.len", "tcp.srcport", "udp.port", "udp.time_delta",
            "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
            "dns.retransmission", "dns.retransmit_request",
            "dns.retransmit_request_in", "mqtt.conflag.cleansess",
            "mqtt.conflags", "mqtt.hdrflags", "mqtt.len", "mqtt.msgtype",
            "mqtt.proto_len", "mqtt.topic_len", "mqtt.ver",
        ],
        "categorical": [
            "http.request.method", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
        ],
    },
    "ton": {
        "path": Path(os.environ.get("DP_SPSR_TON_CSV", "data/ToN-IoT/train_test_network.csv")),
        "label": "label",
        "aux": "type",
        "numeric": [
            "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
            "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
            "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
            "http_trans_depth", "http_request_body_len", "http_response_body_len",
            "http_status_code",
        ],
        "categorical": [
            "proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA",
            "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed",
            "ssl_established", "http_method", "http_version",
            "http_orig_mime_types", "http_resp_mime_types",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the real E6 datasets.")
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--output", type=Path, default=Path("real_e6_data"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _make_strata(label: pd.Series, attack_type: pd.Series) -> pd.Series:
    return label.astype(str) + "::" + attack_type.astype(str)


def _safe_train_test_split(
    df: pd.DataFrame,
    label_col: str,
    aux_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = _make_strata(df[label_col], df[aux_col])
    if strata.value_counts().min() >= 2:
        stratify = strata
    else:
        stratify = df[label_col]
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _make_preprocessor(
    train_frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[ColumnTransformer, list[str], list[str], list[str], pd.DataFrame]:
    numeric_columns = [c for c in numeric_columns if c in train_frame.columns]
    categorical_columns = [c for c in categorical_columns if c in train_frame.columns]
    numeric_frame = train_frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    nonconstant_numeric = [c for c in numeric_columns if numeric_frame[c].nunique(dropna=True) > 1]
    numeric_columns = nonconstant_numeric
    categorical_columns = [c for c in categorical_columns if train_frame[c].nunique(dropna=False) > 1]
    prepared_train = train_frame.copy()
    for column in numeric_columns:
        prepared_train[column] = pd.to_numeric(prepared_train[column], errors="coerce")
    for column in categorical_columns:
        prepared_train[column] = prepared_train[column].astype(str)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns, [], prepared_train


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(name) for name in preprocessor.get_feature_names_out()]


def _transform_frame(
    frame: pd.DataFrame,
    preprocessor: ColumnTransformer,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> np.ndarray:
    work = frame.copy()
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for column in categorical_columns:
        work[column] = work[column].astype(str)
    return np.asarray(preprocessor.transform(work), dtype=np.float64)


def _partition_noniid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    attack_type: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    assignments = [[] for _ in range(n_clients)]
    unique_types = sorted(np.unique(attack_type).tolist())
    normal_types = {"Normal", "normal", "0"}
    for value in unique_types:
        indices = np.flatnonzero(attack_type == value)
        rng.shuffle(indices)
        if str(value) in normal_types:
            proportions = np.full(n_clients, 1.0 / n_clients)
        else:
            proportions = rng.dirichlet(np.full(n_clients, alpha))
        counts = rng.multinomial(len(indices), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            assignments[client_id].extend(indices[start : start + count].tolist())
            start += count
    result = []
    for indices in assignments:
        client_indices = np.asarray(indices, dtype=int)
        rng.shuffle(client_indices)
        result.append(client_indices)
    return result


def preprocess_dataset(
    dataset_name: str,
    config: dict[str, Any],
    output_root: Path,
    test_size: float,
    n_clients: int,
    dirichlet_alpha: float,
    seed: int,
) -> dict[str, Any]:
    path = config["path"]
    frame = pd.read_csv(path, low_memory=False)
    raw_rows = len(frame)
    raw_columns = list(frame.columns)
    exact_duplicate_rows = int(frame.duplicated().sum())
    duplicate_rows_involved = int(frame.duplicated(keep=False).sum())
    frame = frame.drop_duplicates(keep="first").reset_index(drop=True)
    label_col = config["label"]
    aux_col = config["aux"]
    feature_columns = [c for c in frame.columns if c not in {label_col, aux_col}]
    feature_duplicate_rows = int(frame.duplicated(subset=feature_columns).sum())
    conflicting_feature_groups = 0
    grouped = frame.loc[frame.duplicated(subset=feature_columns, keep=False)].groupby(feature_columns, dropna=False)[label_col]
    if len(grouped):
        conflicting_feature_groups = int((grouped.nunique() > 1).sum())
    if conflicting_feature_groups:
        raise ValueError(f"{dataset_name}: conflicting feature-only duplicate groups detected")
    train_frame, test_frame = _safe_train_test_split(frame, label_col, aux_col, test_size, seed)
    preprocessor, numeric_columns, categorical_columns, _, prepared_train = _make_preprocessor(
        train_frame, config["numeric"], config["categorical"]
    )
    preprocessor.fit(prepared_train)
    x_train = _transform_frame(train_frame, preprocessor, numeric_columns, categorical_columns)
    x_test = _transform_frame(test_frame, preprocessor, numeric_columns, categorical_columns)
    y_train = train_frame[label_col].to_numpy(dtype=np.int64)
    y_test = test_frame[label_col].to_numpy(dtype=np.int64)
    type_train = train_frame[aux_col].astype(str).to_numpy()
    type_test = test_frame[aux_col].astype(str).to_numpy()
    client_indices = _partition_noniid(x_train, y_train, type_train, n_clients, dirichlet_alpha, seed)
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "X_train.npy", x_train)
    np.save(dataset_dir / "y_train.npy", y_train)
    np.save(dataset_dir / "type_train.npy", type_train)
    np.save(dataset_dir / "X_test.npy", x_test)
    np.save(dataset_dir / "y_test.npy", y_test)
    np.save(dataset_dir / "type_test.npy", type_test)
    joblib.dump(preprocessor, dataset_dir / "preprocessor.joblib")
    client_summary = []
    for client_id, indices in enumerate(client_indices):
        np.save(dataset_dir / f"client_{client_id}_X.npy", x_train[indices])
        np.save(dataset_dir / f"client_{client_id}_y.npy", y_train[indices])
        np.save(dataset_dir / f"client_{client_id}_type.npy", type_train[indices])
        client_summary.append(
            {
                "client": client_id,
                "samples": int(len(indices)),
                "attack_rate": float(np.mean(y_train[indices])) if len(indices) else None,
                "type_counts": pd.Series(type_train[indices]).value_counts().to_dict(),
            }
        )
    metadata = {
        "dataset": dataset_name,
        "source": str(path),
        "raw_rows": raw_rows,
        "raw_columns": len(raw_columns),
        "deduplicated_rows": int(len(frame)),
        "exact_duplicate_extra_rows": exact_duplicate_rows,
        "exact_duplicate_rows_involved": duplicate_rows_involved,
        "feature_only_duplicate_extra_rows": feature_duplicate_rows,
        "conflicting_feature_duplicate_groups": conflicting_feature_groups,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "feature_dimension": int(x_train.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "encoded_feature_names": _feature_names(preprocessor),
        "label_column": label_col,
        "auxiliary_type_column": aux_col,
        "test_size": test_size,
        "n_clients": n_clients,
        "dirichlet_alpha": dirichlet_alpha,
        "seed": seed,
        "client_summary": client_summary,
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    report = {}
    for name in names:
        report[name] = preprocess_dataset(
            name,
            DATASETS[name],
            output_root,
            args.test_size,
            args.clients,
            args.dirichlet_alpha,
            args.seed,
        )
    (output_root / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8"
    )
    print(f"Saved E6 preprocessing outputs to {output_root}")


if __name__ == "__main__":
    main()
