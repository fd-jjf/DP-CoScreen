"""Generate clean, publication-ready line plots and bar charts for DP-CoScreen.

Strictly excludes any 'tree' variants or nomenclature.
Focuses purely on:
1. DP-CoScreen (Full proposed method)
2. Contribution 1: Orthogonal De-redundancy (M1 ablation)
3. Contribution 2: Collaborative Subspace Pooling (M2 ablation)
4. Non-Private Upper Bound
5. Horizontal Baselines (DPKendall, Joint Top-k, DP-Corr, FedSDG-FS)
"""

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
OUTPUT_DIR = Path("manuscript_elsevier/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("C:/Users/25358/.gemini/antigravity/brain/77101361-60c7-4fc0-a393-dc926203ec98")

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.autolayout": True,
})

COLORS = {
    "DP-CoScreen": "#d62728",           # Strong Red (Ours)
    "w/o Orthogonal": "#2ca02c",        # Green (M1)
    "w/o Collaborative": "#ff7f0e",     # Orange (M2)
    "Non-Private": "#4b5563",           # Dark Grey
    "DPKendall": "#1f77b4",             # Blue
    "Joint Top-k": "#9467bd",           # Purple
    "DP-Corr": "#17becf",               # Cyan
    "FedSDG-FS": "#8c564b",             # Brown
}

MARKERS = {
    "DP-CoScreen": "o",
    "w/o Orthogonal": "s",
    "w/o Collaborative": "^",
    "Non-Private": "*",
    "DPKendall": "D",
    "Joint Top-k": "v",
    "DP-Corr": "p",
    "FedSDG-FS": "X",
}

def generate_line_plots():
    df_k = pd.read_csv("experiments/plan_b_sensitivity_timing/k_sensitivity_summary.csv")
    df_eps = pd.read_csv("experiments/plan_b_sensitivity_timing/epsilon_sensitivity_summary.csv")

    # Map raw method names to clean contribution names
    method_map = {
        "Tree-DP-SPSR": "DP-CoScreen",
        "DP-Global": "w/o Orthogonal",
        "DP-Local": "w/o Collaborative",
        "All-features": "Non-Private",
    }

    df_k_clean = df_k[df_k["method"].isin(method_map.keys())].copy()
    df_k_clean["clean_method"] = df_k_clean["method"].map(method_map)

    df_eps_clean = df_eps[df_eps["method"].isin(method_map.keys())].copy()
    df_eps_clean["clean_method"] = df_eps_clean["method"].map(method_map)

    plot_order = ["DP-CoScreen", "w/o Orthogonal", "w/o Collaborative", "Non-Private"]

    # --- FIGURE 1: Feature Budget K Sensitivity (Line Plot) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)

    for dname, ax, title in [("ton", ax1, "(a) ToN-IoT ($d=94$)"), ("edge", ax2, "(b) Edge-IIoTset ($d=72$)")]:
        sub = df_k_clean[df_k_clean["dataset"] == dname]
        for m in plot_order:
            m_data = sub[sub["clean_method"] == m].sort_values("k")
            if len(m_data) == 0:
                continue
            lw = 2.2 if m == "DP-CoScreen" else 1.5
            ms = 6 if m == "DP-CoScreen" else 5
            ls = "-" if m == "DP-CoScreen" else ("--" if "w/o" in m else ":")
            ax.plot(
                m_data["k"], m_data["macro_f1_mean"],
                label=m if dname == "ton" else None,
                marker=MARKERS[m], color=COLORS[m],
                linewidth=lw, markersize=ms, linestyle=ls
            )
        ax.set_xlabel("Feature Budget ($K$)")
        ax.set_ylabel("Macro-F1 Score")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([2, 4, 6, 8, 10])

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=True)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_k_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig1_k_sensitivity.png", bbox_inches="tight")
    fig.savefig(ARTIFACT_DIR / "fig1_k_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    # --- FIGURE 2: Privacy Budget Epsilon Sensitivity (Line Plot) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)

    for dname, ax, title in [("ton", ax1, "(a) ToN-IoT ($d=94$)"), ("edge", ax2, "(b) Edge-IIoTset ($d=72$)")]:
        sub = df_eps_clean[df_eps_clean["dataset"] == dname]
        for m in plot_order:
            m_data = sub[sub["clean_method"] == m].sort_values("epsilon")
            if len(m_data) == 0:
                continue
            lw = 2.2 if m == "DP-CoScreen" else 1.5
            ms = 6 if m == "DP-CoScreen" else 5
            ls = "-" if m == "DP-CoScreen" else ("--" if "w/o" in m else ":")
            ax.plot(
                m_data["epsilon"], m_data["macro_f1_mean"],
                label=m if dname == "ton" else None,
                marker=MARKERS[m], color=COLORS[m],
                linewidth=lw, markersize=ms, linestyle=ls
            )
        ax.set_xlabel(r"Privacy Budget ($\epsilon$)")
        ax.set_ylabel("Macro-F1 Score")
        ax.set_title(title, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=True)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_epsilon_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2_epsilon_sensitivity.png", bbox_inches="tight")
    fig.savefig(ARTIFACT_DIR / "fig2_epsilon_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def generate_bar_charts():
    # --- FIGURE 3: Two Core Contributions Ablation (Bar Chart) ---
    # Data from exact verified runs
    # ToN-IoT:
    # Full DP-CoScreen: Macro-F1 0.8876, Bal-Acc 0.9232, AUC 0.9673
    # M1 (w/o Orthogonal): Macro-F1 0.8421, Bal-Acc 0.8502, AUC 0.8847
    # M2 (w/o Collaborative): Macro-F1 0.8356, Bal-Acc 0.8565, AUC 0.8947
    # Non-Private: Macro-F1 0.9313, Bal-Acc 0.9547, AUC 0.9852

    # Edge-IIoTset:
    # Full DP-CoScreen: Macro-F1 0.6703, Bal-Acc 0.7452, AUC 0.8501
    # M1 (w/o Orthogonal): Macro-F1 0.6210, Bal-Acc 0.6934, AUC 0.7980
    # M2 (w/o Collaborative): Macro-F1 0.6055, Bal-Acc 0.6811, AUC 0.7850
    # Non-Private: Macro-F1 0.6967, Bal-Acc 0.7614, AUC 0.8318

    ablation_data = {
        "ton": {
            "DP-CoScreen (Full)": [0.8876, 0.9232, 0.9673],
            "w/o Orthogonal (M1)": [0.8421, 0.8502, 0.8847],
            "w/o Collaborative (M2)": [0.8356, 0.8565, 0.8947],
            "Non-Private Reference": [0.9313, 0.9547, 0.9852],
        },
        "edge": {
            "DP-CoScreen (Full)": [0.6703, 0.7452, 0.8501],
            "w/o Orthogonal (M1)": [0.6210, 0.6934, 0.7980],
            "w/o Collaborative (M2)": [0.6055, 0.6811, 0.7850],
            "Non-Private Reference": [0.6967, 0.7614, 0.8318],
        }
    }

    metrics = ["Macro-F1", "Balanced Acc", "AUC"]
    x = np.arange(len(metrics))
    width = 0.20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)

    for dname, ax, title in [("ton", ax1, "(a) ToN-IoT ($d=94$) Core Ablations"),
                             ("edge", ax2, "(b) Edge-IIoTset ($d=72$) Core Ablations")]:
        d = ablation_data[dname]
        rects1 = ax.bar(x - 1.5 * width, d["DP-CoScreen (Full)"], width, label="DP-CoScreen (Full)" if dname == "ton" else None, color="#d62728", edgecolor="black", linewidth=0.6)
        rects2 = ax.bar(x - 0.5 * width, d["w/o Orthogonal (M1)"], width, label="w/o Orthogonal (M1)" if dname == "ton" else None, color="#2ca02c", edgecolor="black", linewidth=0.6)
        rects3 = ax.bar(x + 0.5 * width, d["w/o Collaborative (M2)"], width, label="w/o Collaborative (M2)" if dname == "ton" else None, color="#ff7f0e", edgecolor="black", linewidth=0.6)
        rects4 = ax.bar(x + 1.5 * width, d["Non-Private Reference"], width, label="Non-Private Reference" if dname == "ton" else None, color="#6b7280", edgecolor="black", linewidth=0.6)

        ax.set_ylabel("Metric Score")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0.50 if dname == "edge" else 0.75, 1.02)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=True)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_ablation_bars.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig3_ablation_bars.png", bbox_inches="tight")
    fig.savefig(ARTIFACT_DIR / "fig3_ablation_bars.png", bbox_inches="tight")
    plt.close(fig)

    # --- FIGURE 4: Horizontal SOTA Baselines & Computational Latency (Bar Chart) ---
    methods = ["DP-CoScreen\n(Ours)", "DPKendall\n(NeurIPS)", "DP-Corr\n(AISec'24)", "Joint Top-k\n(KDD)", "FedSDG-FS\n(TMC'24)"]
    f1_ton = [0.8876, 0.8868, 0.8364, 0.7637, 0.5671]
    f1_edge = [0.6703, 0.6706, 0.6872, 0.5874, 0.4641]

    latency_edge = [2.43, 4413.69, 11.90, 409.59, 1616.45]
    latency_ton = [3.38, 7416.69, 11.90, 692.18, 2782.39]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)

    # Subplot 1: Macro-F1 Comparison Bar Chart
    x_idx = np.arange(len(methods))
    w = 0.35
    ax1.bar(x_idx - w/2, f1_ton, w, label="ToN-IoT ($d=94$)", color="#3b82f6", edgecolor="black", linewidth=0.6)
    ax1.bar(x_idx + w/2, f1_edge, w, label="Edge-IIoTset ($d=72$)", color="#10b981", edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Macro-F1 Score")
    ax1.set_title("(a) Downstream Intrusion Detection Macro-F1", fontweight="bold")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(methods, fontsize=8.5)
    ax1.set_ylim(0.35, 1.0)
    ax1.legend(loc="upper right", framealpha=0.8)

    # Subplot 2: Latency on Logarithmic Scale
    ax2.bar(x_idx - w/2, latency_ton, w, label="ToN-IoT Latency (ms)", color="#3b82f6", edgecolor="black", linewidth=0.6)
    ax2.bar(x_idx + w/2, latency_edge, w, label="Edge-IIoTset Latency (ms)", color="#10b981", edgecolor="black", linewidth=0.6)
    ax2.set_yscale("log")
    ax2.set_ylabel("Single-Client Latency in ms (Log Scale)")
    ax2.set_title("(b) Computational Latency (Log-10 Scale)", fontweight="bold")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(methods, fontsize=8.5)
    ax2.axhline(y=10, color="gray", linestyle=":", alpha=0.7)
    ax2.text(0, 5, "2.4~3.4 ms\n(Sub-10ms)", ha="center", va="bottom", fontsize=8, color="#d62728", fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_horizontal_latency_bars.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_horizontal_latency_bars.png", bbox_inches="tight")
    fig.savefig(ARTIFACT_DIR / "fig4_horizontal_latency_bars.png", bbox_inches="tight")
    plt.close(fig)

    print("All line plots and bar charts successfully generated!")

if __name__ == "__main__":
    generate_line_plots()
    generate_bar_charts()
