"""Plan B: Complete Sensitivity and Scalability Benchmark on ToN-IoT & Edge-IIoTset.

Following the IEEE TMC (DP-SFS) evaluation structure:
1. Feature budget sensitivity: k in [2, 4, 6, 8, 10] at epsilon = 1.0
2. Privacy budget sensitivity: epsilon in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0] at k = 10
3. Execution time and scalability: runtime (ms) vs k and runtime (ms) vs epsilon
4. Comprehensive evaluation on 20% Grouped Holdout (GroupShuffleSplit).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dpspsr.core import (
    dp_shared_operator,
    greedy_select,
    leading_subspace,
    shared_structure_from_subspace,
)
from dpspsr.fixed_privacy import (
    FixedFeatureMap,
    StructurePrivacyProtocol,
    empirical_structure,
    symmetric_noise,
)
from dpspsr.task_aware import TaskAwareConfig
from dpspsr.tree_personalization import recover_tree_hierarchical
from preprocess_real_e6 import DATASETS, _partition_noniid


METHODS = ["Tree-DP-SPSR", "Tree-Shared", "DP-Local", "DP-Global", "Shared-only", "All-features"]
K_VALUES = [2, 4, 6, 8, 10]
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
SPLITS = [3101, 3102, 3103, 3104, 3105]
NOISE_SEEDS = [10101, 10102, 10103]


def metric_values(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    pred = (probability >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "attack_recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, probability)),
    }


def fit_and_eval(x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray,
                 subset: np.ndarray) -> dict[str, float]:
    scaler = StandardScaler().fit(x_train[:, subset])
    model = LogisticRegression(solver="liblinear", max_iter=2000,
                               class_weight="balanced", random_state=2026)
    model.fit(scaler.transform(x_train[:, subset]), y_train)
    prob = expit(model.decision_function(scaler.transform(x_test[:, subset])))
    return metric_values(y_test, prob)


def compute_sigma(clip_norm: float, n: int, epsilon: float, delta: float = 1e-5) -> float:
    return float(2 * clip_norm**2 / n / epsilon * np.sqrt(2 * np.log(1.25 / delta)))


def select_variants(h: list[np.ndarray], w: np.ndarray, sigma: np.ndarray, k: int) -> dict[str, list[np.ndarray]]:
    operator = dp_shared_operator(h, w, sigma, lambda_s=0.2)
    p = leading_subspace(operator, 3)
    hs = shared_structure_from_subspace(p, h, w)
    zero = np.zeros_like(h[0])
    pooled = sum(wi * hi for wi, hi in zip(w, h))
    global_subset = greedy_select(pooled, zero, pooled, k, 1, 0, 0.4)[0]
    subsets: dict[str, list[np.ndarray]] = {
        "DP-Local": [],
        "DP-Global": [global_subset] * len(h),
        "Shared-only": [],
    }
    for hi in h:
        subsets["DP-Local"].append(greedy_select(hi, zero, hi, k, 1, 1, 0.4)[0])
        subsets["Shared-only"].append(greedy_select(hs, zero, hi, k, 1, 1, 0.4)[0])
    return subsets


def run_benchmark_on_dataset(
    dataset_name: str,
    output_dir: Path,
    k_list: list[int] = K_VALUES,
    eps_list: list[float] = EPSILON_VALUES,
    splits: list[int] = SPLITS,
    noise_seeds: list[int] = NOISE_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n==================================================")
    print(f"Starting Plan B Benchmark on Dataset: {dataset_name.upper()}")
    print(f"==================================================")

    config = DATASETS[dataset_name]
    data_path = config["path"]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    raw_frame = pd.read_csv(data_path, low_memory=False)
    frame = raw_frame.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"Raw rows: {len(raw_frame)}, Deduplicated rows: {len(frame)}")

    # Grouped 20% holdout split
    group_fields = [c for c in (config["numeric"] + config["categorical"]) if c in frame.columns]
    group_ids = frame.groupby(group_fields, dropna=False, sort=False).ngroup().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, reserved_idx = next(splitter.split(frame, groups=group_ids))
    development = frame.iloc[dev_idx].reset_index(drop=True)
    reserved = frame.iloc[reserved_idx].reset_index(drop=True)
    print(f"Development rows: {len(development)}, Reserved rows: {len(reserved)}")

    # Feature mapping
    mapper = FixedFeatureMap(config["numeric"], config["categorical"], 4)
    x_dev = mapper.transform(development)
    y_dev = development[config["label"]].to_numpy(dtype=int)
    type_dev = development[config["aux"]].astype(str).to_numpy()

    x_res = mapper.transform(reserved)
    y_res = reserved[config["label"]].to_numpy(dtype=int)
    type_res = reserved[config["aux"]].astype(str).to_numpy()
    d_dim = x_dev.shape[1]
    print(f"Feature Dimension: {d_dim}")

    k_metric_records = []
    eps_metric_records = []
    timing_records = []

    with threadpool_limits(limits=1):
        for split_seed in splits:
            print(f"[{dataset_name.upper()}] Processing Split Seed {split_seed}...")
            dev_partitions = _partition_noniid(x_dev, y_dev, type_dev, 4, 0.4, split_seed)
            res_partitions = _partition_noniid(x_res, y_res, type_res, 4, 0.4, split_seed)

            fits, vals = [], []
            for i, idx in enumerate(dev_partitions):
                strata = np.array([f"{l}::{t}" for l, t in zip(y_dev[idx], type_dev[idx])])
                if pd.Series(strata).value_counts().min() < 2:
                    strata = y_dev[idx]
                fit, val = train_test_split(idx, test_size=0.2, random_state=split_seed + i, stratify=strata)
                fits.append(fit)
                vals.append(val)

            # Compute clean empirical second moments
            clean = [empirical_structure(x_dev[idx], y_dev[idx], 7.0) for idx in fits]
            sizes = np.array([len(idx) for idx in fits])
            weights = sizes / sizes.sum()

            # --- PART 1: Varying k at epsilon = 1.0 ---
            protocol_eps1 = StructurePrivacyProtocol(epsilon=1.0, delta=1e-5, clip_norm=7.0, categorical_buckets=4)
            sigmas_eps1 = np.array([protocol_eps1.sigma(int(n)) for n in sizes])

            for noise_seed in noise_seeds:
                noisy_eps1 = [
                    h + symmetric_noise(np.random.default_rng(np.random.SeedSequence([split_seed, noise_seed, i])), len(h), s)
                    for i, (h, s) in enumerate(zip(clean, sigmas_eps1))
                ]

                for k_val in k_list:
                    cfg_k = TaskAwareConfig(k=k_val)
                    common_ridge = max(cfg_k.ridge_floor, np.sqrt(np.sum(weights**2 * sigmas_eps1**2)) * np.sqrt(len(clean[0])))

                    # 1. Tree-DP-SPSR
                    t0 = time.perf_counter()
                    tree = recover_tree_hierarchical(
                        noisy_eps1, sigmas_eps1, weights, cfg_k, shared_fraction=0.5, split_window=1,
                        silhouette_threshold=0.0, max_edge_weight=0.25, fixed_ridge=common_ridge,
                        has_public_intercept=False
                    )
                    t_tree_spsr = (time.perf_counter() - t0) * 1000.0 / len(clean)

                    # 2. Tree-Shared
                    t0 = time.perf_counter()
                    tree_shared = recover_tree_hierarchical(
                        noisy_eps1, sigmas_eps1, weights, cfg_k, shared_fraction=0.5, split_window=1,
                        silhouette_threshold=0.0, max_edge_weight=0.0, fixed_ridge=common_ridge,
                        has_public_intercept=False
                    )
                    t_tree_shared = (time.perf_counter() - t0) * 1000.0 / len(clean)

                    # 3. Baselines: DP-Local, DP-Global, Shared-only
                    t0 = time.perf_counter()
                    old_subsets = select_variants(noisy_eps1, weights, sigmas_eps1, k=k_val)
                    t_variants = (time.perf_counter() - t0) * 1000.0 / len(clean)

                    methods_dict = {
                        "Tree-DP-SPSR": (tree["selected_sets"], t_tree_spsr),
                        "Tree-Shared": (tree_shared["selected_sets"], t_tree_shared),
                        "DP-Local": (old_subsets["DP-Local"], t_variants),
                        "DP-Global": (old_subsets["DP-Global"], t_variants),
                        "Shared-only": (old_subsets["Shared-only"], t_variants),
                        "All-features": ([np.arange(d_dim)] * 4, 0.0),
                    }

                    # Evaluate on reserved test set
                    for client_i, (fit_idx, res_idx) in enumerate(zip(fits, res_partitions)):
                        for m_name, (subsets_list, t_ms) in methods_dict.items():
                            sel_features = np.asarray(subsets_list[client_i], dtype=int)
                            res_metrics = fit_and_eval(x_dev[fit_idx], y_dev[fit_idx], x_res[res_idx], y_res[res_idx], sel_features)
                            k_metric_records.append({
                                "dataset": dataset_name,
                                "split_seed": split_seed,
                                "noise_seed": noise_seed,
                                "client": client_i,
                                "k": k_val,
                                "epsilon": 1.0,
                                "method": m_name,
                                "time_ms": t_ms,
                                **res_metrics,
                            })
                            timing_records.append({
                                "dataset": dataset_name,
                                "experiment": "varying_k",
                                "param": k_val,
                                "method": m_name,
                                "time_ms": t_ms,
                            })

            # --- PART 2: Varying epsilon at k = 10 ---
            for eps_val in eps_list:
                sigmas_eps = np.array([compute_sigma(7.0, int(n), eps_val, 1e-5) for n in sizes])

                for noise_seed in noise_seeds:
                    noisy_eps = [
                        h + symmetric_noise(np.random.default_rng(np.random.SeedSequence([split_seed, noise_seed, i])), len(h), s)
                        for i, (h, s) in enumerate(zip(clean, sigmas_eps))
                    ]
                    cfg_10 = TaskAwareConfig(k=10)
                    common_ridge = max(cfg_10.ridge_floor, np.sqrt(np.sum(weights**2 * sigmas_eps**2)) * np.sqrt(len(clean[0])))

                    t0 = time.perf_counter()
                    tree = recover_tree_hierarchical(
                        noisy_eps, sigmas_eps, weights, cfg_10, shared_fraction=0.5, split_window=1,
                        silhouette_threshold=0.0, max_edge_weight=0.25, fixed_ridge=common_ridge,
                        has_public_intercept=False
                    )
                    t_tree_spsr = (time.perf_counter() - t0) * 1000.0 / len(clean)

                    t0 = time.perf_counter()
                    tree_shared = recover_tree_hierarchical(
                        noisy_eps, sigmas_eps, weights, cfg_10, shared_fraction=0.5, split_window=1,
                        silhouette_threshold=0.0, max_edge_weight=0.0, fixed_ridge=common_ridge,
                        has_public_intercept=False
                    )
                    t_tree_shared = (time.perf_counter() - t0) * 1000.0 / len(clean)

                    t0 = time.perf_counter()
                    old_subsets = select_variants(noisy_eps, weights, sigmas_eps, k=10)
                    t_variants = (time.perf_counter() - t0) * 1000.0 / len(clean)

                    methods_dict = {
                        "Tree-DP-SPSR": (tree["selected_sets"], t_tree_spsr),
                        "Tree-Shared": (tree_shared["selected_sets"], t_tree_shared),
                        "DP-Local": (old_subsets["DP-Local"], t_variants),
                        "DP-Global": (old_subsets["DP-Global"], t_variants),
                        "Shared-only": (old_subsets["Shared-only"], t_variants),
                        "All-features": ([np.arange(d_dim)] * 4, 0.0),
                    }

                    for client_i, (fit_idx, res_idx) in enumerate(zip(fits, res_partitions)):
                        for m_name, (subsets_list, t_ms) in methods_dict.items():
                            sel_features = np.asarray(subsets_list[client_i], dtype=int)
                            res_metrics = fit_and_eval(x_dev[fit_idx], y_dev[fit_idx], x_res[res_idx], y_res[res_idx], sel_features)
                            eps_metric_records.append({
                                "dataset": dataset_name,
                                "split_seed": split_seed,
                                "noise_seed": noise_seed,
                                "client": client_i,
                                "k": 10,
                                "epsilon": eps_val,
                                "method": m_name,
                                "time_ms": t_ms,
                                **res_metrics,
                            })
                            timing_records.append({
                                "dataset": dataset_name,
                                "experiment": "varying_eps",
                                "param": eps_val,
                                "method": m_name,
                                "time_ms": t_ms,
                            })

    df_k = pd.DataFrame(k_metric_records)
    df_eps = pd.DataFrame(eps_metric_records)
    df_timing = pd.DataFrame(timing_records)
    return df_k, df_eps, df_timing


def generate_plots(output_dir: Path, df_k_all: pd.DataFrame, df_eps_all: pd.DataFrame, df_time_all: pd.DataFrame):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })

    markers = {"Tree-DP-SPSR": "o", "Tree-Shared": "s", "DP-Local": "^",
               "DP-Global": "v", "Shared-only": "D", "All-features": "*"}
    colors = {"Tree-DP-SPSR": "#d62728", "Tree-Shared": "#1f77b4", "DP-Local": "#ff7f0e",
              "DP-Global": "#2ca02c", "Shared-only": "#9467bd", "All-features": "#7f7f7f"}

    datasets = df_k_all["dataset"].unique()

    # --- FIGURE 1: Utility vs k ---
    for dname in datasets:
        sub_k = df_k_all[df_k_all["dataset"] == dname]
        mean_k = sub_k.groupby(["k", "method"])[["macro_f1", "balanced_accuracy"]].mean().reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        for m in METHODS:
            m_data = mean_k[mean_k["method"] == m]
            if len(m_data) == 0:
                continue
            ax1.plot(m_data["k"], m_data["macro_f1"], label=m, marker=markers.get(m, "o"),
                     color=colors.get(m, "blue"), linewidth=1.5, markersize=5)
            ax2.plot(m_data["k"], m_data["balanced_accuracy"], label=m, marker=markers.get(m, "o"),
                     color=colors.get(m, "blue"), linewidth=1.5, markersize=5)

        ax1.set_xlabel("Feature Budget ($k$)")
        ax1.set_ylabel("Macro-F1")
        ax1.set_title(f"({dname.upper()}) Macro-F1 vs. Feature Budget $k$")
        ax1.legend(loc="lower right", framealpha=0.8, fontsize=8)

        ax2.set_xlabel("Feature Budget ($k$)")
        ax2.set_ylabel("Balanced Accuracy")
        ax2.set_title(f"({dname.upper()}) Balanced Accuracy vs. Feature Budget $k$")
        ax2.legend(loc="lower right", framealpha=0.8, fontsize=8)

        plt.tight_layout()
        fig.savefig(output_dir / f"fig1_k_sensitivity_{dname}.pdf")
        fig.savefig(output_dir / f"fig1_k_sensitivity_{dname}.png")
        plt.close(fig)

    # --- FIGURE 2: Utility vs epsilon ---
    for dname in datasets:
        sub_eps = df_eps_all[df_eps_all["dataset"] == dname]
        mean_eps = sub_eps.groupby(["epsilon", "method"])[["macro_f1", "balanced_accuracy"]].mean().reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        for m in METHODS:
            m_data = mean_eps[mean_eps["method"] == m]
            if len(m_data) == 0:
                continue
            ax1.plot(m_data["epsilon"], m_data["macro_f1"], label=m, marker=markers.get(m, "o"),
                     color=colors.get(m, "blue"), linewidth=1.5, markersize=5)
            ax2.plot(m_data["epsilon"], m_data["balanced_accuracy"], label=m, marker=markers.get(m, "o"),
                     color=colors.get(m, "blue"), linewidth=1.5, markersize=5)

        ax1.set_xlabel(r"Privacy Budget ($\epsilon$)")
        ax1.set_ylabel("Macro-F1")
        ax1.set_title(rf"({dname.upper()}) Macro-F1 vs. Privacy Budget $\epsilon$")
        ax1.set_xscale("log")
        ax1.legend(loc="lower right", framealpha=0.8, fontsize=8)

        ax2.set_xlabel(r"Privacy Budget ($\epsilon$)")
        ax2.set_ylabel("Balanced Accuracy")
        ax2.set_title(rf"({dname.upper()}) Balanced Accuracy vs. Privacy Budget $\epsilon$")
        ax2.set_xscale("log")
        ax2.legend(loc="lower right", framealpha=0.8, fontsize=8)

        plt.tight_layout()
        fig.savefig(output_dir / f"fig2_epsilon_sensitivity_{dname}.pdf")
        fig.savefig(output_dir / f"fig2_epsilon_sensitivity_{dname}.png")
        plt.close(fig)

    # --- FIGURE 3: Runtime Scalability ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    time_k = df_time_all[df_time_all["experiment"] == "varying_k"].groupby(["param", "method"])["time_ms"].mean().reset_index()
    for m in ["Tree-DP-SPSR", "Tree-Shared", "DP-Local", "DP-Global", "Shared-only"]:
        m_data = time_k[time_k["method"] == m]
        if len(m_data) == 0:
            continue
        ax1.plot(m_data["param"], m_data["time_ms"], label=m, marker=markers.get(m, "o"),
                 color=colors.get(m, "blue"), linewidth=1.5, markersize=5)

    ax1.set_xlabel("Feature Budget ($k$)")
    ax1.set_ylabel("Selection Runtime (ms)")
    ax1.set_title("Runtime vs. Feature Budget $k$")
    ax1.legend(loc="upper left", framealpha=0.8, fontsize=8)

    time_eps = df_time_all[df_time_all["experiment"] == "varying_eps"].groupby(["param", "method"])["time_ms"].mean().reset_index()
    for m in ["Tree-DP-SPSR", "Tree-Shared", "DP-Local", "DP-Global", "Shared-only"]:
        m_data = time_eps[time_eps["method"] == m]
        if len(m_data) == 0:
            continue
        ax2.plot(m_data["param"], m_data["time_ms"], label=m, marker=markers.get(m, "o"),
                 color=colors.get(m, "blue"), linewidth=1.5, markersize=5)

    ax2.set_xlabel(r"Privacy Budget ($\epsilon$)")
    ax2.set_ylabel("Selection Runtime (ms)")
    ax2.set_title(r"Runtime vs. Privacy Budget $\epsilon$")
    ax2.set_xscale("log")
    ax2.legend(loc="upper right", framealpha=0.8, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_runtime_scalability.pdf")
    fig.savefig(output_dir / "fig3_runtime_scalability.png")
    plt.close(fig)
    print("All plots successfully generated in:", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ton", "edge", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("experiments/plan_b_sensitivity_timing"))
    parser.add_argument("--splits", type=int, nargs="+", default=SPLITS)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=NOISE_SEEDS)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_datasets = ["ton", "edge"] if args.dataset == "all" else [args.dataset]

    k_dfs, eps_dfs, time_dfs = [], [], []
    # If partial dataset requested and existing outputs exist, keep the other dataset's rows
    if args.dataset != "all":
        raw_k_path = out_dir / "k_sensitivity_raw.csv"
        raw_eps_path = out_dir / "epsilon_sensitivity_raw.csv"
        raw_time_path = out_dir / "timing_raw.csv"
        if raw_k_path.exists() and raw_eps_path.exists() and raw_time_path.exists():
            old_k = pd.read_csv(raw_k_path)
            old_eps = pd.read_csv(raw_eps_path)
            old_time = pd.read_csv(raw_time_path)
            other_name = "ton" if args.dataset == "edge" else "edge"
            k_dfs.append(old_k[old_k["dataset"] == other_name])
            eps_dfs.append(old_eps[old_eps["dataset"] == other_name])
            time_dfs.append(old_time[old_time["dataset"] == other_name])

    for dname in target_datasets:
        df_k, df_eps, df_time = run_benchmark_on_dataset(
            dname, out_dir, splits=args.splits, noise_seeds=args.noise_seeds
        )
        k_dfs.append(df_k)
        eps_dfs.append(df_eps)
        time_dfs.append(df_time)

    combined_k = pd.concat(k_dfs, ignore_index=True)
    combined_eps = pd.concat(eps_dfs, ignore_index=True)
    combined_time = pd.concat(time_dfs, ignore_index=True)

    # Save CSVs
    combined_k.to_csv(out_dir / "k_sensitivity_raw.csv", index=False)
    combined_eps.to_csv(out_dir / "epsilon_sensitivity_raw.csv", index=False)
    combined_time.to_csv(out_dir / "timing_raw.csv", index=False)

    # Summaries
    metrics = ["macro_f1", "attack_recall", "balanced_accuracy", "auc", "time_ms"]
    k_summary = combined_k.groupby(["dataset", "method", "k"])[metrics].agg(["mean", "std"]).reset_index()
    k_summary.columns = ["_".join(filter(None, c)) for c in k_summary.columns]
    k_summary.to_csv(out_dir / "k_sensitivity_summary.csv", index=False)

    eps_summary = combined_eps.groupby(["dataset", "method", "epsilon"])[metrics].agg(["mean", "std"]).reset_index()
    eps_summary.columns = ["_".join(filter(None, c)) for c in eps_summary.columns]
    eps_summary.to_csv(out_dir / "epsilon_sensitivity_summary.csv", index=False)

    time_summary = combined_time.groupby(["dataset", "experiment", "method", "param"])["time_ms"].agg(["mean", "std"]).reset_index()
    time_summary.to_csv(out_dir / "timing_summary.csv", index=False)

    # Generate Figures
    generate_plots(out_dir, combined_k, combined_eps, combined_time)
    print("\nBenchmark complete! All results and figures saved to:", out_dir)


if __name__ == "__main__":
    main()
