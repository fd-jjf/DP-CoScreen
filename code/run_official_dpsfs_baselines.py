"""Benchmark DP-SFS and its official baselines (DPKendall, Joint, FedSDG-FS) vs DP-SPSR.

Runs directly on the ToN-IoT & Edge-IIoTset 20% Grouped Holdout protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from threadpoolctl import threadpool_limits

# Add paths
sys.path.insert(0, str(Path("code").resolve()))
sys.path.insert(0, str(Path("reference_repos/DP-SFS").resolve()))

from dp_sfs import dp_spectral_feature_selection
from benchmark_methods import dpkendall_selection, joint_topk, fedsdg_selection

from dpspsr.fixed_privacy import (
    FixedFeatureMap,
    StructurePrivacyProtocol,
    empirical_structure,
    symmetric_noise,
)
from dpspsr.task_aware import TaskAwareConfig
from dpspsr.tree_personalization import recover_tree_hierarchical
from preprocess_real_e6 import DATASETS, _partition_noniid
from run_plan_b_sensitivity_timing import fit_and_eval


def run_benchmark(dataset_name: str = "ton", n_splits: int = 3, k: int = 10, epsilon: float = 1.0, seed_start: int = 3101):
    config = DATASETS[dataset_name]
    raw = pd.read_csv(config["path"], low_memory=False).drop_duplicates(keep="first").reset_index(drop=True)
    
    group_fields = [c for c in (config["numeric"] + config["categorical"]) if c in raw.columns]
    group_ids = raw.groupby(group_fields, dropna=False, sort=False).ngroup().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, res_idx = next(splitter.split(raw, groups=group_ids))

    dev_frame = raw.iloc[dev_idx].reset_index(drop=True)
    res_frame = raw.iloc[res_idx].reset_index(drop=True)

    mapper = FixedFeatureMap(config["numeric"], config["categorical"], 4)
    x_dev = mapper.transform(dev_frame)
    y_dev = dev_frame[config["label"]].to_numpy(dtype=int)
    type_dev = dev_frame[config["aux"]].astype(str).to_numpy()

    x_res = mapper.transform(res_frame)
    y_res = res_frame[config["label"]].to_numpy(dtype=int)
    type_res = res_frame[config["aux"]].astype(str).to_numpy()

    splits = [seed_start + i for i in range(n_splits)]
    records = []

    print(f"\n=======================================================")
    print(f"Running Official Baselines Benchmark on {dataset_name.upper()} (k={k}, eps={epsilon})")
    print(f"=======================================================")

    with threadpool_limits(limits=1):
        for s_idx, split_seed in enumerate(splits):
            print(f"[{dataset_name.upper()}] Processing Split {s_idx+1}/{n_splits} (Seed {split_seed})...")
            dev_parts = _partition_noniid(x_dev, y_dev, type_dev, 4, 0.4, split_seed)
            res_parts = _partition_noniid(x_res, y_res, type_res, 4, 0.4, split_seed)

            fits = []
            for i, idx in enumerate(dev_parts):
                fit, _ = train_test_split(idx, test_size=0.2, random_state=split_seed + i)
                fits.append(fit)

            # Tree-DP-SPSR preparation
            clean = [empirical_structure(x_dev[idx], y_dev[idx], 7.0) for idx in fits]
            sizes = np.array([len(idx) for idx in fits])
            weights = sizes / sizes.sum()
            protocol = StructurePrivacyProtocol(epsilon=epsilon, delta=1e-5, clip_norm=7.0, categorical_buckets=4)
            sigmas = np.array([protocol.sigma(int(n)) for n in sizes])
            noisy = [
                h + symmetric_noise(np.random.default_rng(np.random.SeedSequence([split_seed, 10101, i])), len(h), s)
                for i, (h, s) in enumerate(zip(clean, sigmas))
            ]

            cfg = TaskAwareConfig(k=k)
            ridge = max(cfg.ridge_floor, np.sqrt(np.sum(weights**2 * sigmas**2)) * np.sqrt(len(clean[0])))
            
            t0 = time.perf_counter()
            tree = recover_tree_hierarchical(
                noisy, sigmas, weights, cfg, shared_fraction=0.5, split_window=1,
                silhouette_threshold=0.0, max_edge_weight=0.25, fixed_ridge=ridge,
                has_public_intercept=False
            )
            t_spsr = (time.perf_counter() - t0) * 1000.0 / len(clean)

            # Run client-wise evaluations
            for c_i in range(4):
                xc = x_dev[fits[c_i]]
                yc = y_dev[fits[c_i]]
                xr = x_res[res_parts[c_i]]
                yr = y_res[res_parts[c_i]]

                # 1. Tree-DP-SPSR
                sel_spsr = np.array(tree["selected_sets"][c_i], dtype=int)
                m_spsr = fit_and_eval(xc, yc, xr, yr, sel_spsr)
                records.append({
                    "dataset": dataset_name, "split": split_seed, "client": c_i,
                    "method": "Tree-DP-SPSR", "time_ms": t_spsr, **m_spsr
                })

                # 2. DP-SFS
                t0 = time.perf_counter()
                sel_sfs = dp_spectral_feature_selection(xc, yc, k=k, epsilon=epsilon, seed=split_seed * 10 + c_i)
                t_sfs = (time.perf_counter() - t0) * 1000.0
                m_sfs = fit_and_eval(xc, yc, xr, yr, sel_sfs)
                records.append({
                    "dataset": dataset_name, "split": split_seed, "client": c_i,
                    "method": "DP-SFS", "time_ms": t_sfs, **m_sfs
                })

                # 3. DPKendall
                t0 = time.perf_counter()
                sel_kendall = dpkendall_selection(xc, yc, k=k, epsilon=epsilon)
                t_kendall = (time.perf_counter() - t0) * 1000.0
                m_kendall = fit_and_eval(xc, yc, xr, yr, sel_kendall)
                records.append({
                    "dataset": dataset_name, "split": split_seed, "client": c_i,
                    "method": "DPKendall", "time_ms": t_kendall, **m_kendall
                })

                # 4. Joint
                t0 = time.perf_counter()
                sel_joint = joint_topk(xc, yc, k=k, epsilon=epsilon)
                t_joint = (time.perf_counter() - t0) * 1000.0
                m_joint = fit_and_eval(xc, yc, xr, yr, sel_joint)
                records.append({
                    "dataset": dataset_name, "split": split_seed, "client": c_i,
                    "method": "Joint", "time_ms": t_joint, **m_joint
                })

                # 5. FedSDG-FS
                t0 = time.perf_counter()
                sel_fedsdg = fedsdg_selection(xc, yc, k=k, epsilon=epsilon, n_iters=50, seed=split_seed * 10 + c_i)
                t_fedsdg = (time.perf_counter() - t0) * 1000.0
                m_fedsdg = fit_and_eval(xc, yc, xr, yr, sel_fedsdg)
                records.append({
                    "dataset": dataset_name, "split": split_seed, "client": c_i,
                    "method": "FedSDG-FS", "time_ms": t_fedsdg, **m_fedsdg
                })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    out_dir = Path("experiments/official_baselines_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_ton = run_benchmark("ton", n_splits=3, k=10, epsilon=1.0)
    df_ton.to_csv(out_dir / "official_baselines_ton_raw.csv", index=False)
    
    summary = df_ton.groupby("method")[["macro_f1", "attack_recall", "balanced_accuracy", "auc", "time_ms"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.sort_values(by="macro_f1_mean", ascending=False)
    summary.to_csv(out_dir / "official_baselines_ton_summary.csv")
    
    print("\n================== BENCHMARK SUMMARY (ToN-IoT, 20% Grouped Holdout) ==================")
    print(summary[["macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "auc_mean", "time_ms_mean"]])
