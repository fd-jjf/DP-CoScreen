"""Component ablation on the frozen grouped ToN-IoT holdout protocol.

All private variants reuse the same per-client noisy second-moment release.
The non-private tree is a development-only reference and is labelled as such.
The reserved partition is used only for final client-local evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dpspsr.fixed_privacy import (
    FixedFeatureMap, StructurePrivacyProtocol, empirical_structure, symmetric_noise,
)
from dpspsr.task_aware import TaskAwareConfig, conditional_select_blocks, psd_projection
from dpspsr.tree_personalization import recover_tree_hierarchical
from preprocess_real_e6 import DATASETS, _partition_noniid
from run_fixed_protocol_pilot import sha256, variant_subsets


METHODS = {
    "Full-DP-SPSR": "full tree-guided private selector",
    "No-branch-personalization": "tree branch residual weight fixed to zero",
    "No-conditional-redundancy": "rank by marginal task relevance without conditional redundancy control",
    "No-tree-flat-personalization": "client-specific shrunken task profiles without hierarchical grouping",
    "No-cross-client-sharing": "independent DP-Local selector",
    "Pruned-correlated-features": "full selector followed by fit-fold |r| >= 0.95 pruning without replacement",
    "No-DP-noise (non-private)": "same tree selector on clean development structures",
}


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    pred = (probability >= .5).astype(int)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "attack_recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, probability)),
    }


def fit_reserved(x_fit: np.ndarray, y_fit: np.ndarray, x_test: np.ndarray,
                 y_test: np.ndarray, subset: np.ndarray) -> dict[str, float]:
    scaler = StandardScaler().fit(x_fit[:, subset])
    model = LogisticRegression(
        solver="liblinear", max_iter=2000, class_weight="balanced", random_state=2026,
    )
    model.fit(scaler.transform(x_fit[:, subset]), y_fit)
    probability = expit(model.decision_function(scaler.transform(x_test[:, subset])))
    return metric_values(y_test, probability)


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.sum(centered**2, axis=0))
    normalized = np.divide(
        centered, scale, out=np.zeros_like(centered), where=scale[None, :] > 1e-12,
    )
    return np.clip(normalized.T @ normalized, -1, 1)


def selection_redundancy(x: np.ndarray, subset: np.ndarray) -> float:
    if len(subset) < 2:
        return 0.0
    corr = np.abs(correlation_matrix(x[:, subset]))
    return float(corr[np.triu_indices(len(subset), 1)].mean())


def prune_correlated(x: np.ndarray, subset: np.ndarray,
                     threshold: float = .95) -> np.ndarray:
    """Keep the earlier selected feature when a later one is highly correlated."""
    corr = np.abs(correlation_matrix(x[:, subset]))
    kept_positions: list[int] = []
    for position in range(len(subset)):
        if all(corr[position, previous] < threshold for previous in kept_positions):
            kept_positions.append(position)
    return np.asarray(subset, dtype=int)[kept_positions]


def selected_pairs(x: np.ndarray, subset: np.ndarray, feature_names: tuple[str, ...],
                   split: int, noise_seed: int, client: int) -> list[dict]:
    corr = np.abs(correlation_matrix(x[:, subset]))
    rows = []
    for left, right in combinations(range(len(subset)), 2):
        a, b = int(subset[left]), int(subset[right])
        if a > b:
            a, b = b, a
        rows.append({
            "split_seed": split, "noise_seed": noise_seed, "client": client,
            "feature_a": feature_names[a], "feature_b": feature_names[b],
            "feature_a_index": a, "feature_b_index": b,
            "absolute_correlation": float(corr[left, right]),
        })
    return rows


def mean_client_jaccard(selected_sets: list[np.ndarray]) -> float:
    values = []
    for left, right in combinations(selected_sets, 2):
        a, b = set(map(int, left)), set(map(int, right))
        values.append(len(a & b) / len(a | b))
    return float(np.mean(values))


def flat_personalized(noisy: list[np.ndarray], sigmas: np.ndarray,
                      weights: np.ndarray, config: TaskAwareConfig,
                      ridge: float, max_weight: float = .25) -> list[np.ndarray]:
    pool = sum(w * h for w, h in zip(weights, noisy))
    a = psd_projection(pool[:-1, :-1])
    q = pool[:-1, -1]
    beta = np.linalg.solve(a + ridge * np.eye(len(q)), q)
    profiles = [h[:-1, -1] - h[:-1, :-1] @ beta for h in noisy]
    variances = [np.maximum(
        sigma**2 * (.5 * (1 + float(beta @ beta)) + .5 * beta**2), 1e-12,
    ) for sigma in sigmas]
    mean_profile = sum(w * profile for w, profile in zip(weights, profiles))
    pooled_variance = sum(w**2 * variance for w, variance in zip(weights, variances))
    selected = []
    for i, profile in enumerate(profiles):
        delta = profile - mean_profile
        contrast_variance = np.maximum(
            variances[i] + pooled_variance - 2 * weights[i] * variances[i], 0,
        )
        energy = float(delta @ delta)
        reliability = max(0., 1 - float(contrast_variance.sum()) / energy) if energy > 0 else 0.
        task = q + max_weight * reliability * delta
        subset, _ = conditional_select_blocks(a, task, config.k, ridge, conditional=True)
        selected.append(subset)
    return selected


def paired_rows(repeat: pd.DataFrame, endpoint: str) -> list[dict]:
    pivot = repeat.pivot(index="split_seed", columns="method", values=endpoint)
    rows = []
    for comparator in METHODS:
        if comparator == "Full-DP-SPSR":
            continue
        delta = pivot["Full-DP-SPSR"] - pivot[comparator]
        radius = t.ppf(.975, len(delta)-1) * delta.std(ddof=1) / np.sqrt(len(delta))
        rows.append({
            "endpoint": endpoint, "method": "Full-DP-SPSR", "comparator": comparator,
            "paired_delta": float(delta.mean()), "ci95_low": float(delta.mean()-radius),
            "ci95_high": float(delta.mean()+radius),
            "positive_splits": int((delta > 0).sum()), "splits": int(len(delta)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent

    config = dict(DATASETS["ton"])
    config["path"] = args.data.resolve()
    protocol = StructurePrivacyProtocol(
        epsilon=1, delta=1e-5, clip_norm=7, categorical_buckets=4,
    )
    mapper = FixedFeatureMap(config["numeric"], config["categorical"], 4)
    split_seeds = list(range(3101, 3106))
    noise_seeds = list(range(10101, 10104))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "dataset": "ToN-IoT", "dataset_path": str(config["path"]),
        "dataset_sha256": sha256(config["path"]), "privacy": protocol.manifest(),
        "development_split": "grouped exact retained-input pattern; seed 42; 80/20",
        "split_seeds": split_seeds, "noise_seeds": noise_seeds,
        "clients": 4, "dirichlet_alpha": .4, "client_validation_fraction": .2,
        "parameters": {"k": 10, "shared_fraction": .5, "max_edge_weight": .25,
                       "split_window": 1, "silhouette_threshold": 0.,
                       "ridge_floor": .001, "noise_ridge_multiplier": 1.,
                       "correlation_pruning_threshold": .95},
        "methods": METHODS, "feature_map": mapper.manifest(),
        "statistical_unit": "split seed; average clients and paired noise seeds first",
        "reserved_use": "evaluation only; no release, recovery, selection, or tuning",
        "feature_redundancy": "mean absolute Pearson correlation among selected mapped features on each development fit fold",
        "sources": {name: sha256(root / name) for name in [
            "run_grouped_component_ablation.py", "run_fixed_protocol_pilot.py",
            "dpspsr/fixed_privacy.py", "dpspsr/task_aware.py",
            "dpspsr/tree_personalization.py", "preprocess_real_e6.py",
        ]},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    raw = pd.read_csv(config["path"], low_memory=False)
    frame = raw.drop_duplicates(keep="first").reset_index(drop=True)
    group_fields = config["numeric"] + config["categorical"]
    groups = frame.groupby(group_fields, dropna=False, sort=False).ngroup().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    dev_idx, reserved_idx = next(splitter.split(frame, groups=groups))
    development = frame.iloc[dev_idx].reset_index(drop=True)
    reserved = frame.iloc[reserved_idx].reset_index(drop=True)
    x_dev = mapper.transform(development)
    y_dev = development[config["label"]].to_numpy(dtype=int)
    type_dev = development[config["aux"]].astype(str).to_numpy()
    x_res = mapper.transform(reserved)
    y_res = reserved[config["label"]].to_numpy(dtype=int)
    type_res = reserved[config["aux"]].astype(str).to_numpy()
    data_info = {
        "raw_rows": int(len(raw)), "deduplicated_rows": int(len(frame)),
        "development_rows": int(len(development)), "reserved_rows": int(len(reserved)),
        "groups": int(np.unique(groups).size),
        "development_sha256": hashlib.sha256(development.to_csv(index=False).encode()).hexdigest(),
        "reserved_sha256": hashlib.sha256(reserved.to_csv(index=False).encode()).hexdigest(),
    }

    metric_rows, selection_rows, jaccard_rows, pair_rows = [], [], [], []
    cfg = TaskAwareConfig(k=10)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for split in split_seeds:
            dev_parts = _partition_noniid(x_dev, y_dev, type_dev, 4, .4, split)
            res_parts = _partition_noniid(x_res, y_res, type_res, 4, .4, split)
            fits = []
            for i, indices in enumerate(dev_parts):
                strata = np.array([
                    f"{label}::{typ}" for label, typ in zip(y_dev[indices], type_dev[indices])
                ])
                if pd.Series(strata).value_counts().min() < 2:
                    strata = y_dev[indices]
                fit, _ = train_test_split(
                    indices, test_size=.2, random_state=split+i, stratify=strata,
                )
                fits.append(fit)
            clean = [empirical_structure(x_dev[idx], y_dev[idx], protocol.clip_norm) for idx in fits]
            sizes = np.array([len(idx) for idx in fits])
            weights = sizes / sizes.sum()
            sigmas = np.array([protocol.sigma(int(n)) for n in sizes])
            no_dp = recover_tree_hierarchical(
                clean, np.zeros(4), weights, cfg, shared_fraction=.5, split_window=1,
                silhouette_threshold=0., max_edge_weight=.25, fixed_ridge=cfg.ridge_floor,
                has_public_intercept=False,
            )["selected_sets"]

            for noise_seed in noise_seeds:
                noisy = [
                    h + symmetric_noise(
                        np.random.default_rng(np.random.SeedSequence([split, noise_seed, i])),
                        len(h), sigma,
                    )
                    for i, (h, sigma) in enumerate(zip(clean, sigmas))
                ]
                common_ridge = max(
                    cfg.ridge_floor,
                    np.sqrt(np.sum(weights**2 * sigmas**2)) * np.sqrt(len(clean[0])),
                )
                full = recover_tree_hierarchical(
                    noisy, sigmas, weights, cfg, shared_fraction=.5, split_window=1,
                    silhouette_threshold=0., max_edge_weight=.25, fixed_ridge=common_ridge,
                    has_public_intercept=False,
                )["selected_sets"]
                no_branch = recover_tree_hierarchical(
                    noisy, sigmas, weights, cfg, shared_fraction=.5, split_window=1,
                    silhouette_threshold=0., max_edge_weight=0., fixed_ridge=common_ridge,
                    has_public_intercept=False,
                )["selected_sets"]
                no_redundancy = recover_tree_hierarchical(
                    noisy, sigmas, weights, cfg, shared_fraction=.5, split_window=1,
                    silhouette_threshold=0., max_edge_weight=.25, fixed_ridge=common_ridge,
                    has_public_intercept=False, conditional=False,
                )["selected_sets"]
                flat = flat_personalized(noisy, sigmas, weights, cfg, common_ridge)
                legacy, _, _ = variant_subsets(noisy, weights, sigmas)
                pruned = [
                    prune_correlated(x_dev[fit], np.asarray(subset, dtype=int), threshold=.95)
                    for fit, subset in zip(fits, full)
                ]
                selected_by_method = {
                    "Full-DP-SPSR": full,
                    "No-branch-personalization": no_branch,
                    "No-conditional-redundancy": no_redundancy,
                    "No-tree-flat-personalization": flat,
                    "No-cross-client-sharing": legacy["DP-Local"],
                    "Pruned-correlated-features": pruned,
                    "No-DP-noise (non-private)": no_dp,
                }
                for method, selected_sets in selected_by_method.items():
                    jaccard_rows.append({
                        "split_seed": split, "noise_seed": noise_seed, "method": method,
                        "client_selection_jaccard": mean_client_jaccard(selected_sets),
                    })
                    for i, (fit, res_idx) in enumerate(zip(fits, res_parts)):
                        selected = np.asarray(selected_sets[i], dtype=int)
                        values = fit_reserved(
                            x_dev[fit], y_dev[fit], x_res[res_idx], y_res[res_idx], selected,
                        )
                        redundancy = selection_redundancy(x_dev[fit], selected)
                        metric_rows.append({
                            "split_seed": split, "noise_seed": noise_seed, "client": i,
                            "method": method, "test_rows": int(len(res_idx)),
                            "selected_k": int(len(selected)),
                            "selection_redundancy": redundancy, **values,
                        })
                        selection_rows.append({
                            "split_seed": split, "noise_seed": noise_seed, "client": i,
                            "method": method, "selected": json.dumps(selected.tolist()),
                            "selected_k": int(len(selected)),
                            "selected_names": json.dumps(
                                [mapper.feature_names[j] for j in selected], ensure_ascii=False,
                            ),
                            "selection_redundancy": redundancy,
                        })
                        if method == "Full-DP-SPSR":
                            pair_rows.extend(selected_pairs(
                                x_dev[fit], selected, mapper.feature_names, split, noise_seed, i,
                            ))
            print(f"split={split} complete", flush=True)

    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    jaccard = pd.DataFrame(jaccard_rows)
    metrics.to_csv(out / "per_client_metrics.csv", index=False)
    selections.to_csv(out / "selections.csv", index=False)
    jaccard.to_csv(out / "client_selection_jaccard.csv", index=False)

    endpoints = [
        "macro_f1", "attack_recall", "balanced_accuracy", "auc",
        "selection_redundancy", "selected_k",
    ]
    repeat = metrics.groupby(["split_seed", "method"], as_index=False)[endpoints].mean()
    repeat_jaccard = jaccard.groupby(["split_seed", "method"], as_index=False)[
        "client_selection_jaccard"
    ].mean()
    repeat = repeat.merge(repeat_jaccard, on=["split_seed", "method"], validate="one_to_one")
    repeat.to_csv(out / "repeat_summary.csv", index=False)
    summary = repeat.groupby("method")[endpoints + ["client_selection_jaccard"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    paired = pd.DataFrame(
        paired_rows(repeat, "macro_f1") + paired_rows(repeat, "selection_redundancy")
        + paired_rows(repeat, "selected_k")
    )
    paired.to_csv(out / "paired_differences.csv", index=False)

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "full_selected_feature_pair_correlations.csv", index=False)
    pair_summary = pairs.groupby(
        ["feature_a_index", "feature_b_index", "feature_a", "feature_b"], as_index=False,
    ).agg(
        selection_count=("absolute_correlation", "size"),
        mean_absolute_correlation=("absolute_correlation", "mean"),
        max_absolute_correlation=("absolute_correlation", "max"),
        high_correlation_count=("absolute_correlation", lambda values: int((values >= .8).sum())),
    )
    pair_summary["high_correlation_rate"] = (
        pair_summary.high_correlation_count / pair_summary.selection_count
    )
    pair_summary = pair_summary.sort_values(
        ["mean_absolute_correlation", "selection_count"], ascending=False,
    )
    pair_summary.to_csv(out / "feature_pair_summary.csv", index=False)
    candidates = pair_summary[
        (pair_summary.mean_absolute_correlation >= .8)
        | ((pair_summary.high_correlation_rate >= .5) & (pair_summary.selection_count >= 5))
    ]
    candidates.to_csv(out / "redundant_feature_pairs.csv", index=False)

    report = {
        "status": "complete", "seconds": time.perf_counter()-started,
        "data": data_info, "metric_rows": int(len(metrics)),
        "expected_metric_rows": 5*3*4*len(METHODS),
        "feature_pair_rows": int(len(pairs)),
        "redundant_feature_pair_candidates": int(len(candidates)),
        "final_holdout_evaluated": True, "source_sha256": manifest["sources"],
    }
    report["metric_rows_complete"] = report["metric_rows"] == report["expected_metric_rows"]
    if not report["metric_rows_complete"]:
        raise AssertionError("Incomplete ablation metric table")
    (out / "run_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    manifest.update(status="complete", results=report, finished_utc=datetime.now(timezone.utc).isoformat())
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({"output": str(out), **report}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
