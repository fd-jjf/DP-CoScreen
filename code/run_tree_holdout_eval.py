"""Strict reserved-holdout evaluation for the frozen tree-guided protocol.

The 80% development split is the only source used for client construction,
structure release, tree recovery, feature selection and classifier fitting.
The 20% reserved split is partitioned only for reporting client-local test
metrics and is never used to select features or tune a parameter.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scipy.special import expit

from dpspsr.fixed_privacy import FixedFeatureMap, StructurePrivacyProtocol, empirical_structure, symmetric_noise
from dpspsr.task_aware import TaskAwareConfig
from dpspsr.tree_personalization import recover_tree_hierarchical
from preprocess_real_e6 import DATASETS, _partition_noniid, _safe_train_test_split
from run_fixed_protocol_pilot import evaluate, sha256, variant_subsets


METHODS = ["Tree-DP-SPSR", "Tree-Shared", "DP-Local", "DP-Global", "Shared-only", "All-features"]


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    pred = (probability >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "attack_recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, probability)),
    }


def fit_reserved_test(x_fit: np.ndarray, y_fit: np.ndarray,
                      x_test: np.ndarray, y_test: np.ndarray,
                      subset: np.ndarray) -> dict[str, float]:
    scaler = StandardScaler().fit(x_fit[:, subset])
    model = LogisticRegression(solver="liblinear", max_iter=2000,
                               class_weight="balanced", random_state=2026)
    model.fit(scaler.transform(x_fit[:, subset]), y_fit)
    probability = expit(model.decision_function(scaler.transform(x_test[:, subset])))
    return metric_values(y_test, probability)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="path to the ToN-IoT CSV")
    parser.add_argument(
        "--grouped-input-holdout", action="store_true",
        help="split at the exact retained-input-field group level",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    config = DATASETS["ton"]
    config["path"] = args.data.resolve()
    protocol = StructurePrivacyProtocol(epsilon=1, delta=1e-5, clip_norm=7,
                                        categorical_buckets=4)
    mapper = FixedFeatureMap(config["numeric"], config["categorical"], 4)
    split_seeds, noise_seeds = list(range(3101, 3106)), list(range(10101, 10104))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "dataset": "ToN-IoT", "dataset_path": str(config["path"]),
        "dataset_sha256": sha256(config["path"]), "privacy": protocol.manifest(),
        "development_split_seed": 42, "reserved_fraction": 0.2,
        "split_seeds": split_seeds, "noise_seeds": noise_seeds,
        "clients": 4, "dirichlet_alpha": 0.4, "client_validation_fraction": 0.2,
        "parameters": {"k": 10, "shared_fraction": 0.5, "max_edge_weight": 0.25,
                       "split_window": 1, "silhouette_threshold": 0.0, "ridge_floor": 0.001,
                       "noise_ridge_multiplier": 1.0},
        "feature_map": mapper.manifest(),
        "reserved_use": "reserved rows are partitioned for evaluation only; no release, tree, selection or tuning uses them",
        "reserved_partition": "same _partition_noniid rule and each split seed, independently applied to reserved rows",
        "classifier": "local StandardScaler + balanced liblinear logistic regression, threshold 0.5",
        "statistical_unit": "split seed; average clients and paired noise seeds first",
        "sources": {name: sha256(root / name) for name in [
            "run_tree_holdout_eval.py", "run_tree_fixed_protocol.py", "run_fixed_protocol_pilot.py",
            "dpspsr/fixed_privacy.py", "dpspsr/task_aware.py",
            "dpspsr/tree_personalization.py", "preprocess_real_e6.py"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    raw_frame = pd.read_csv(config["path"], low_memory=False)
    raw_rows = len(raw_frame)
    frame = raw_frame.drop_duplicates(keep="first").reset_index(drop=True)
    if args.grouped_input_holdout:
        group_fields = config["numeric"] + config["categorical"]
        # Grouping is performed before feature mapping and excludes label/type,
        # so an exact retained-input pattern cannot cross the evaluation split.
        group_ids = frame.groupby(group_fields, dropna=False, sort=False).ngroup().to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        dev_idx, reserved_idx = next(splitter.split(frame, groups=group_ids))
        development = frame.iloc[dev_idx].reset_index(drop=True)
        reserved = frame.iloc[reserved_idx].reset_index(drop=True)
        split_protocol = "grouped exact retained-input pattern split; GroupShuffleSplit seed 42"
    else:
        development, reserved = _safe_train_test_split(frame, config["label"], config["aux"], .2, 42)
        split_protocol = "record-level stratified split seed 42"
    x_dev, y_dev = mapper.transform(development), development[config["label"]].to_numpy(dtype=int)
    type_dev = development[config["aux"]].astype(str).to_numpy()
    x_res, y_res = mapper.transform(reserved), reserved[config["label"]].to_numpy(dtype=int)
    type_res = reserved[config["aux"]].astype(str).to_numpy()
    data_info = {
        "raw_rows": int(raw_rows), "deduplicated_rows": int(len(frame)),
        "development_rows": int(len(development)), "reserved_test_rows": int(len(reserved)),
        "reserved_used": True,
        "development_reserved_split": split_protocol,
        "grouped_input_fields": (config["numeric"] + config["categorical"]
                                  if args.grouped_input_holdout else []),
        "grouped_input_unique_groups": int(np.unique(group_ids).size)
        if args.grouped_input_holdout else None,
        "development_label_rate": float(y_dev.mean()), "reserved_label_rate": float(y_res.mean()),
        "development_sha256": hashlib.sha256(development.to_csv(index=False).encode()).hexdigest(),
        "reserved_sha256": hashlib.sha256(reserved.to_csv(index=False).encode()).hexdigest(),
    }
    metric_rows, selection_rows = [], []
    started = time.perf_counter()

    with threadpool_limits(limits=1):
        for split in split_seeds:
            dev_partitions = _partition_noniid(x_dev, y_dev, type_dev, 4, .4, split)
            res_partitions = _partition_noniid(x_res, y_res, type_res, 4, .4, split)
            fits, vals = [], []
            for i, idx in enumerate(dev_partitions):
                strata = np.array([f"{label}::{typ}" for label, typ in zip(y_dev[idx], type_dev[idx])])
                if pd.Series(strata).value_counts().min() < 2:
                    strata = y_dev[idx]
                fit, val = train_test_split(idx, test_size=.2, random_state=split+i, stratify=strata)
                fits.append(fit); vals.append(val)
            clean = [empirical_structure(x_dev[idx], y_dev[idx], protocol.clip_norm) for idx in fits]
            sizes = np.array([len(idx) for idx in fits])
            weights = sizes / sizes.sum()
            sigmas = np.array([protocol.sigma(int(n)) for n in sizes])
            for noise_seed in noise_seeds:
                noisy = [h + symmetric_noise(
                    np.random.default_rng(np.random.SeedSequence([split, noise_seed, i])), len(h), sigma)
                    for i, (h, sigma) in enumerate(zip(clean, sigmas))]
                cfg = TaskAwareConfig(k=10)
                old_subsets, _, _ = variant_subsets(noisy, weights, sigmas)
                common_ridge = max(cfg.ridge_floor,
                    np.sqrt(np.sum(weights**2 * sigmas**2)) * np.sqrt(len(clean[0])))
                tree = recover_tree_hierarchical(
                    noisy, sigmas, weights, cfg, shared_fraction=.5, split_window=1,
                    silhouette_threshold=0., max_edge_weight=.25,
                    fixed_ridge=common_ridge, has_public_intercept=False)
                tree_shared = recover_tree_hierarchical(
                    noisy, sigmas, weights, cfg, shared_fraction=.5, split_window=1,
                    silhouette_threshold=0., max_edge_weight=0.,
                    fixed_ridge=common_ridge, has_public_intercept=False)
                selected_by_method = {
                    "Tree-DP-SPSR": tree["selected_sets"], "Tree-Shared": tree_shared["selected_sets"],
                    "DP-Local": old_subsets["DP-Local"], "DP-Global": old_subsets["DP-Global"],
                    "Shared-only": old_subsets["Shared-only"],
                    "All-features": [np.arange(x_dev.shape[1])] * 4,
                }
                for i, (fit, res_idx) in enumerate(zip(fits, res_partitions)):
                    for method in METHODS:
                        selected = np.asarray(selected_by_method[method][i], dtype=int)
                        values_local = fit_reserved_test(x_dev[fit], y_dev[fit], x_res[res_idx], y_res[res_idx], selected)
                        values_pooled = fit_reserved_test(x_dev[fit], y_dev[fit], x_res, y_res, selected)
                        for view, indices, values in [("reserved_local", res_idx, values_local),
                                                      ("reserved_pooled", np.arange(len(x_res)), values_pooled)]:
                            metric_rows.append(dict(dataset="ton", split_seed=split,
                                noise_seed=noise_seed, client=i, method=method, view=view,
                                test_rows=int(len(indices)), **values))
                        selection_rows.append(dict(dataset="ton", split_seed=split,
                            noise_seed=noise_seed, client=i, method=method,
                            selected=json.dumps(selected.tolist()), selected_k=len(selected),
                            personal_weight=(tree["diagnostics"][i]["personal_weight"]
                                             if method == "Tree-DP-SPSR" else np.nan),
                            final_tree_clusters=(tree["diagnostics"][i]["final_cluster_count"]
                                                 if method == "Tree-DP-SPSR" else np.nan)))
            print(f"split={split} complete", flush=True)

    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    metrics.to_csv(out / "reserved_per_client_metrics.csv", index=False)
    selections.to_csv(out / "development_selections.csv", index=False)
    metric_names = ["macro_f1", "attack_recall", "balanced_accuracy", "auc"]
    repeat = metrics.groupby(["dataset", "split_seed", "method", "view"], as_index=False)[metric_names].mean()
    repeat.to_csv(out / "reserved_repeat_summary.csv", index=False)
    summary = repeat.groupby(["dataset", "method", "view"])[metric_names].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "reserved_summary.csv", index=False)
    pairs = []
    local_repeat = repeat[repeat.view.eq("reserved_local")]
    means = local_repeat.pivot(index="split_seed", columns="method", values="macro_f1")
    for comparator in ["Tree-Shared", "DP-Local", "DP-Global", "Shared-only", "All-features"]:
        delta = means["Tree-DP-SPSR"] - means[comparator]
        radius = t.ppf(.975, len(delta)-1) * delta.std(ddof=1) / np.sqrt(len(delta))
        pairs.append({"endpoint": "reserved_local_macro_f1", "method": "Tree-DP-SPSR",
                      "comparator": comparator, "paired_delta": float(delta.mean()),
                      "ci95_low": float(delta.mean()-radius), "ci95_high": float(delta.mean()+radius),
                      "positive_splits": int((delta > 0).sum()), "splits": len(delta)})
    pd.DataFrame(pairs).to_csv(out / "reserved_paired_differences.csv", index=False)
    report = {"status": "complete", "seconds": time.perf_counter()-started,
              "data": data_info, "metric_rows": len(metrics),
              "expected_metric_rows": 5*3*4*6*2, "final_holdout_evaluated": True,
              "source_sha256": manifest["sources"]}
    report["metric_rows_complete"] = report["metric_rows"] == report["expected_metric_rows"]
    if not report["metric_rows_complete"]:
        raise AssertionError("Incomplete holdout metric table")
    (out / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest.update(status="complete", results=report, finished_utc=datetime.now(timezone.utc).isoformat())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), **report}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
