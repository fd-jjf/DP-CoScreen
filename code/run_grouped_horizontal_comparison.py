"""Classical feature-selection comparison on the paper's grouped ToN-IoT holdout.

All selectors use development-client fit records only. The reserved input-group
holdout is used only for evaluation. Classical selectors are non-private
references and are not presented as DP methods.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from threadpoolctl import threadpool_limits

from dpspsr.fixed_privacy import FixedFeatureMap
from horizontal_baselines import select_federated_chi2, select_mrmr_mid
from preprocess_real_e6 import DATASETS, _partition_noniid, _safe_train_test_split
from run_tree_holdout_eval import fit_reserved_test


SPLITS = tuple(range(3101, 3106))
TREE_RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "holdout_grouped_input_20261025"
METRICS = ("macro_f1", "attack_recall", "balanced_accuracy", "auc")
METHODS = (
    "Local-MI (non-DP)",
    "Local-F (non-DP)",
    "Local-MI-Corr (non-DP)",
    "Fed-FeRe-inspired Chi2 (non-DP)",
    "Centralized mRMR-MID (non-DP)",
    "Random-K (non-DP)",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_local(x: np.ndarray, y: np.ndarray, method: str, seed: int,
                 discrete: np.ndarray, k: int) -> np.ndarray:
    if method == "Local-MI (non-DP)":
        scores = mutual_info_classif(x, y, discrete_features=discrete,
                                     random_state=seed, n_neighbors=3)
    elif method == "Local-F (non-DP)":
        scores, _ = f_classif(x, y)
    elif method == "Local-MI-Corr (non-DP)":
        relevance = mutual_info_classif(x, y, discrete_features=discrete,
                                        random_state=seed, n_neighbors=3)
        corr = np.nan_to_num(np.abs(np.corrcoef(x, rowvar=False)), nan=0.0,
                             posinf=0.0, neginf=0.0)
        selected: list[int] = []
        remaining = set(range(x.shape[1]))
        for _ in range(k):
            best = max(remaining, key=lambda j: (
                relevance[j] - (float(corr[j, selected].mean()) if selected else 0.0), -j
            ))
            selected.append(best)
            remaining.remove(best)
        return np.asarray(selected, dtype=int)
    elif method == "Random-K (non-DP)":
        return np.random.default_rng(seed).choice(x.shape[1], size=k, replace=False)
    else:
        raise ValueError(method)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max,
                           neginf=-np.inf)
    return np.argsort(-scores, kind="stable")[:k]


def grouped_split(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_fields = config["numeric"] + config["categorical"]
    groups = frame.groupby(group_fields, dropna=False, sort=False).ngroup().to_numpy()
    dev_idx, reserved_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=.2, random_state=42
    ).split(frame, groups=groups))
    return (frame.iloc[dev_idx].reset_index(drop=True),
            frame.iloc[reserved_idx].reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("shuju/train_test_network.csv"))
    parser.add_argument("--tree-results", type=Path, default=TREE_RESULTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    config = DATASETS["ton"]
    mapper = FixedFeatureMap(config["numeric"], config["categorical"], 4)
    raw = pd.read_csv(args.data, low_memory=False)
    raw_rows = len(raw)
    frame = raw.drop_duplicates(keep="first").reset_index(drop=True)
    dev, reserved = grouped_split(frame, config)
    x_dev, y_dev = mapper.transform(dev), dev[config["label"]].to_numpy(dtype=int)
    y_type = dev[config["aux"]].astype(str).to_numpy()
    x_reserved = mapper.transform(reserved)
    y_reserved = reserved[config["label"]].to_numpy(dtype=int)
    reserved_type = reserved[config["aux"]].astype(str).to_numpy()
    feature_names = list(mapper.feature_names)
    discrete = np.asarray([name.startswith(("missing::", "hash::"))
                           for name in feature_names], dtype=bool)

    metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for split in SPLITS:
            dev_clients = _partition_noniid(x_dev, y_dev, y_type, 4, .4, split)
            reserved_clients = _partition_noniid(
                x_reserved, y_reserved, reserved_type, 4, .4, split
            )
            fit_x, fit_y = [], []
            for client, ids in enumerate(dev_clients):
                strata = np.asarray([f"{label}::{kind}"
                                     for label, kind in zip(y_dev[ids], y_type[ids])])
                if pd.Series(strata).value_counts().min() < 2:
                    strata = y_dev[ids]
                fit_ids, _ = train_test_split(
                    ids, test_size=.2, random_state=split + client, stratify=strata
                )
                fit_x.append(x_dev[fit_ids])
                fit_y.append(y_dev[fit_ids])

            selected: dict[str, list[np.ndarray]] = {}
            for method in METHODS:
                if method == "Fed-FeRe-inspired Chi2 (non-DP)":
                    shared = select_federated_chi2(fit_x, fit_y, args.k).indices
                    selected[method] = [shared.copy() for _ in fit_x]
                elif method == "Centralized mRMR-MID (non-DP)":
                    pooled_x = np.vstack(fit_x)
                    pooled_y = np.concatenate(fit_y)
                    shared = select_mrmr_mid(
                        pooled_x, pooled_y, args.k, feature_names, bins=10
                    ).indices
                    selected[method] = [shared.copy() for _ in fit_x]
                else:
                    selected[method] = [
                        select_local(x, y, method, split + client, discrete, args.k)
                        for client, (x, y) in enumerate(zip(fit_x, fit_y))
                    ]

            for client, (x_fit, y_fit, test_ids) in enumerate(
                    zip(fit_x, fit_y, reserved_clients)):
                for method in METHODS:
                    subset = selected[method][client]
                    values = fit_reserved_test(
                        x_fit, y_fit, x_reserved[test_ids], y_reserved[test_ids], subset
                    )
                    metric_rows.append({
                        "dataset": "ton", "split_seed": split, "client": client,
                        "method": method, "view": "reserved_local",
                        "test_rows": len(test_ids), **values,
                    })
                    selection_rows.append({
                        "dataset": "ton", "split_seed": split, "client": client,
                        "method": method, "selected": " ".join(map(str, subset.tolist())),
                        "selected_k": len(subset), "privacy": "non-private",
                    })
            print(f"split={split} complete", flush=True)

    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    metrics.to_csv(out / "baseline_per_client_metrics.csv", index=False)
    selections.to_csv(out / "baseline_selections.csv", index=False)
    repeat = metrics.groupby(["dataset", "split_seed", "method", "view"],
                             as_index=False)[list(METRICS)].mean()
    repeat.to_csv(out / "baseline_repeat_summary.csv", index=False)
    summary = repeat.groupby(["dataset", "method", "view"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(out / "baseline_summary.csv", index=False)

    tree_path = args.tree_results / "reserved_repeat_summary.csv"
    tree = pd.read_csv(tree_path)
    tree = tree[(tree["view"] == "reserved_local") &
                (tree["method"].isin(["Tree-DP-SPSR", "Tree-Shared", "DP-Local",
                                      "DP-Global", "Shared-only", "All-features"]))]
    all_repeat = pd.concat([tree, repeat], ignore_index=True)
    all_repeat.to_csv(out / "combined_repeat_summary.csv", index=False)
    all_summary = all_repeat.groupby(["dataset", "method", "view"])[list(METRICS)].agg(["mean", "std"])
    all_summary.columns = ["_".join(col) for col in all_summary.columns]
    all_summary.reset_index().to_csv(out / "combined_summary.csv", index=False)

    paired = all_repeat.pivot(index="split_seed", columns="method", values="macro_f1")
    pairs = []
    for method in [*METHODS, "Tree-Shared", "DP-Local", "DP-Global",
                   "Shared-only", "All-features"]:
        delta = paired["Tree-DP-SPSR"] - paired[method]
        radius = t.ppf(.975, len(delta)-1) * delta.std(ddof=1) / np.sqrt(len(delta))
        pairs.append({"method": "Tree-DP-SPSR", "comparator": method,
                      "paired_macro_f1_delta": float(delta.mean()),
                      "ci95_low": float(delta.mean()-radius),
                      "ci95_high": float(delta.mean()+radius),
                      "positive_splits": int((delta > 0).sum()), "splits": len(delta)})
    pd.DataFrame(pairs).to_csv(out / "paired_macro_f1_differences.csv", index=False)

    report = {
        "status": "complete", "dataset": "ToN-IoT",
        "dataset_sha256": sha256(args.data), "tree_results": str(tree_path),
        "raw_rows": raw_rows, "deduplicated_rows": len(frame),
        "development_rows": len(dev), "reserved_rows": len(reserved),
        "reserved_used_for_selection": False, "grouped_holdout_seed": 42,
        "split_seeds": list(SPLITS), "clients": 4, "dirichlet_alpha": .4,
        "features": x_dev.shape[1], "selected_k": args.k,
        "methods": list(METHODS), "method_privacy": "all new classical references are non-private",
        "classifier": "local StandardScaler + balanced liblinear logistic regression, threshold 0.5",
        "metric_rows": len(metrics), "expected_metric_rows": len(SPLITS)*4*len(METHODS),
        "seconds": time.perf_counter()-started,
    }
    report["metric_rows_complete"] = report["metric_rows"] == report["expected_metric_rows"]
    (out / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        **report, "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": sha256(args.data),
        "protocol": "same exact-input grouped development/reserved split and client partition seeds as final DP-SPSR holdout; selectors fit on development client fit folds only",
        "sources": {name: sha256(Path(__file__).with_name(name)) for name in [
            "run_grouped_horizontal_comparison.py", "horizontal_baselines.py",
            "run_tree_holdout_eval.py", "preprocess_real_e6.py"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not report["metric_rows_complete"]:
        raise AssertionError("Incomplete baseline metrics")
    print(json.dumps({"output": str(out), **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
