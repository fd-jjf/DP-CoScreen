"""Tree-guided rerun of the frozen ToN-IoT development protocol.

All methods share one [x, y, 1] DP release per client. The reserved test split
is excluded. Results are paired against four fixed controls.
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
from threadpoolctl import threadpool_limits
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from dpspsr.fixed_privacy import (FixedFeatureMap, StructurePrivacyProtocol,
                                  empirical_structure, symmetric_noise)
from dpspsr.task_aware import TaskAwareConfig
from dpspsr.tree_personalization import recover_tree_hierarchical
from preprocess_real_e6 import DATASETS, _partition_noniid, _safe_train_test_split
from run_fixed_protocol_pilot import evaluate, sha256, variant_subsets


METHODS = ["Tree-DP-SPSR", "Tree-Shared", "DP-Local", "DP-Global", "Shared-only", "All-features"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="path to the ToN-IoT CSV")
    parser.add_argument("--grouped-input-holdout", action="store_true",
                        help="use exact retained-input groups for the development split")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    protocol = StructurePrivacyProtocol(epsilon=1, delta=1e-5, clip_norm=7,
                                        categorical_buckets=4)
    split_seeds, noise_seeds = list(range(3101, 3106)), list(range(10101, 10104))
    config = DATASETS["ton"]
    config["path"] = args.data.resolve()
    mapper = FixedFeatureMap(config["numeric"], config["categorical"], 4)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), status="running",
        dataset="ToN-IoT", dataset_path=str(config["path"]),
        dataset_sha256=sha256(config["path"]),
        privacy=protocol.manifest(), split_seeds=split_seeds,
        noise_seeds=noise_seeds, clients=4, dirichlet_alpha=.4,
        reserved_test_split_seed=42, reserved_test_fraction=.2,
        client_validation_fraction=.2, final_holdout_transformed_or_evaluated=False,
        statistic="[fixed-map x, signed binary y] second moment (same release layout as 2026-10-22 pilot)",
        protocol_comparison_note=("Same raw ToN-IoT data, fixed map, split/noise seeds, privacy budget, "
            "client construction, classifier and four comparator implementations as the 2026-10-22 "
            "pilot. Tree recovery uses a no-centering compatibility mode because the frozen release "
            "contains no public intercept coordinate."),
        parameters=dict(k=10, shared_fraction=.5, max_edge_weight=.25,
                        split_window=1, silhouette_threshold=0., ridge_floor=.001,
                        noise_ridge_multiplier=1.),
        methods=dict(
            **{"Tree-DP-SPSR": "shared 50% prefix + one nested DP-profile tree + capped branch residual on the original release"},
            **{"Tree-Shared": "same tree recovery and shared prefix as Tree-DP-SPSR, with branch residual weight fixed to zero"},
            **{"DP-Local": "independent client task-aware feature selection from each client's noisy release"},
            **{"DP-Global": "one shared subset from the weighted pooled noisy release"},
            **{"Shared-only": "shared task-aware subset, no client branch personalization"},
            **{"All-features": "all 94 fixed-map coordinates; nonprivate reference"}),
        feature_map=mapper.manifest(),
        statistical_unit="split seed; average clients and paired noise seeds first",
        interval_scope="exploratory paired t intervals over five overlapping-data development repartitions",
        classifier="local StandardScaler + balanced liblinear logistic regression, threshold 0.5",
        sources={name: sha256(root / name) for name in [
            "run_tree_fixed_protocol.py", "run_fixed_protocol_pilot.py",
            "dpspsr/fixed_privacy.py", "dpspsr/task_aware.py",
            "dpspsr/tree_personalization.py", "preprocess_real_e6.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    frame = pd.read_csv(config["path"], low_memory=False).drop_duplicates(keep="first").reset_index(drop=True)
    if args.grouped_input_holdout:
        group_fields = config["numeric"] + config["categorical"]
        group_ids = frame.groupby(group_fields, dropna=False, sort=False).ngroup().to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        dev_idx, reserved_idx = next(splitter.split(frame, groups=group_ids))
        development = frame.iloc[dev_idx].reset_index(drop=True)
        reserved = frame.iloc[reserved_idx].reset_index(drop=True)
        split_protocol = "grouped exact retained-input pattern split; GroupShuffleSplit seed 42"
    else:
        development, reserved = _safe_train_test_split(frame, config["label"], config["aux"], .2, 42)
        split_protocol = "record-level stratified split seed 42"
    data_info = dict(raw_rows=int(pd.read_csv(config["path"], usecols=[config["label"]]).shape[0]),
                     deduplicated_rows=len(frame), development_rows=len(development),
                     reserved_test_rows=len(reserved), reserved_used=False,
                     development_reserved_split=split_protocol)
    del frame, reserved
    x = mapper.transform(development)
    y = development[config["label"]].to_numpy(dtype=int)
    attack_type = development[config["aux"]].astype(str).to_numpy()
    metric_rows, selection_rows = [], []
    started = time.perf_counter()

    with threadpool_limits(limits=1):
        for split in split_seeds:
            partitions = _partition_noniid(x, y, attack_type, 4, .4, split)
            fits, vals = [], []
            for i, idx in enumerate(partitions):
                strata = np.array([f"{label}::{typ}" for label, typ in zip(y[idx], attack_type[idx])])
                if pd.Series(strata).value_counts().min() < 2:
                    strata = y[idx]
                fit, val = train_test_split(idx, test_size=.2, random_state=split+i, stratify=strata)
                fits.append(fit); vals.append(val)
            all_fit, all_val = np.concatenate(fits), np.concatenate(vals)
            if len(np.intersect1d(all_fit, all_val)) or len(all_fit)+len(all_val) != len(y):
                raise AssertionError("Invalid client fit/validation partition")
            clean = [empirical_structure(x[idx], y[idx], protocol.clip_norm) for idx in fits]
            sizes = np.array([len(idx) for idx in fits])
            weights = sizes / sizes.sum()
            sigmas = np.array([protocol.sigma(int(n)) for n in sizes])
            pooled_validation_x, pooled_validation_y = x[all_val], y[all_val]
            cache = {}
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
                    "Tree-DP-SPSR": tree["selected_sets"],
                    "Tree-Shared": tree_shared["selected_sets"],
                    "DP-Local": old_subsets["DP-Local"],
                    "DP-Global": old_subsets["DP-Global"],
                    "Shared-only": old_subsets["Shared-only"],
                    "All-features": [np.arange(x.shape[1])] * 4,
                }
                for i, (fit, val) in enumerate(zip(fits, vals)):
                    for method in METHODS:
                        selected = np.asarray(selected_by_method[method][i], dtype=int)
                        key = (i, tuple(sorted(selected.tolist())))
                        if key not in cache:
                            cache[key] = evaluate(x[fit], y[fit], x[val], y[val],
                                pooled_validation_x, pooled_validation_y, selected)
                        for view, values in cache[key].items():
                            metric_rows.append(dict(dataset="ton", split_seed=split,
                                noise_seed=noise_seed, client=i, method=method,
                                view=view, **values))
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
    metrics.to_csv(out / "per_client_metrics.csv", index=False)
    selections.to_csv(out / "selections.csv", index=False)
    metric_names = ["macro_f1", "attack_recall", "balanced_accuracy", "auc"]
    repeat = metrics.groupby(["dataset", "split_seed", "method", "view"], as_index=False)[metric_names].mean()
    repeat.to_csv(out / "repeat_summary.csv", index=False)
    summary = repeat.groupby(["dataset", "method", "view"])[metric_names].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    pairs = []
    local_means = repeat[repeat.view.eq("local")].pivot(index="split_seed", columns="method", values="macro_f1")
    for comparator in ["Tree-Shared", "DP-Local", "DP-Global", "Shared-only", "All-features"]:
        delta = local_means["Tree-DP-SPSR"] - local_means[comparator]
        radius = t.ppf(.975, len(delta)-1) * delta.std(ddof=1) / np.sqrt(len(delta))
        pairs.append(dict(dataset="ton", endpoint="local_validation_macro_f1",
            method="Tree-DP-SPSR", comparator=comparator, paired_delta=float(delta.mean()),
            ci95_low=float(delta.mean()-radius), ci95_high=float(delta.mean()+radius),
            positive_splits=int((delta > 0).sum()), splits=len(delta)))
    pd.DataFrame(pairs).to_csv(out / "paired_differences.csv", index=False)
    report = dict(status="complete", seconds=time.perf_counter()-started,
        data=data_info, metric_rows=len(metrics), expected_metric_rows=5*3*4*6*2,
        final_holdout_evaluated=False, source_sha256=manifest["sources"])
    report["metric_rows_complete"] = report["metric_rows"] == report["expected_metric_rows"]
    if not report["metric_rows_complete"]:
        raise AssertionError("Incomplete metric table")
    (out / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    manifest.update(status="complete", results=report, finished_utc=datetime.now(timezone.utc).isoformat())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps({"output": str(out), **report}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
