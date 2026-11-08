"""Generate Nature-style Framework Diagram for DP-CoScreen.

Design principles (Nature journal standards):
- Modular panel structure with bold lowercase letter tags: a, b, c, d
- Visual data representations:
  * Raw data matrix grid (X) and target vector (y)
  * Empirical covariance heatmap before DP
  * Privatized covariance heatmap after Gaussian noise
  * Collaborative pooling from M clients with 1/M variance cancellation badge
  * Geometric / residual projection for Schur complement orthogonal screening
  * Selected feature circles and high-speed edge IDS deployment
- Nature Publishing Group (NPG) color palette
- Clean sans-serif typography, zero text clipping or overlapping
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("manuscript_elsevier/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("C:/Users/25358/.gemini/antigravity/brain/77101361-60c7-4fc0-a393-dc926203ec98")

# Nature-style Typography
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
})

def draw_nature_framework():
    fig = plt.figure(figsize=(15.5, 7.6), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#ffffff")

    # -------------------------------------------------------------
    # TOP HEADER
    # -------------------------------------------------------------
    ax.text(50, 96.5, "DP-CoScreen: Differentially Private Collaborative Orthogonal Screening Framework",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#1e293b")
    ax.text(50, 93.8, "Non-IID Edge Telemetry  •  One-Shot Gaussian Release  •  1/M Subspace Noise Cancellation  •  Schur Complement Projection",
            ha="center", va="center", fontsize=10, style="italic", color="#64748b")
    ax.plot([5, 95], [91.5, 91.5], color="#cbd5e1", linewidth=1.2)

    # -------------------------------------------------------------
    # PANEL A: Stage 1 - Local One-Shot DP Statistics Release (Clients C_1 ... C_M)
    # -------------------------------------------------------------
    # Card Background
    card_a = patches.FancyBboxPatch((3.5, 7), 28, 81, boxstyle="round,pad=1.2",
                                    facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5)
    ax.add_patch(card_a)

    # Header Bar
    hdr_a = patches.FancyBboxPatch((3.5, 83), 28, 5, boxstyle="round,pad=0.5",
                                   facecolor="#e2e8f0", edgecolor="#94a3b8", linewidth=1.2)
    ax.add_patch(hdr_a)
    ax.text(5.5, 85.5, "a", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(8.5, 85.5, "Stage 1: Edge Gateways", fontsize=11.5, fontweight="bold", color="#1e293b")
    ax.text(28.5, 85.5, "Clients $C_1 \\dots C_M$", fontsize=9, style="italic", color="#64748b", ha="right")

    # 1. Visual Data Matrix
    ax.text(17.5, 80.5, "Raw Telemetry Stream $D_i = \\{(x_{it}, y_{it})\\}$", ha="center", fontsize=9.5, fontweight="bold", color="#334155")
    
    # Draw Matrix Grid (7x7 features + 1 target vector)
    start_x, start_y = 7.5, 67.5
    cell_s = 1.05
    np.random.seed(42)
    for r in range(6):
        for c in range(6):
            val = np.random.rand()
            color = "#f1f5f9" if val > 0.35 else "#cbd5e1"
            rect = patches.Rectangle((start_x + c * cell_s, start_y - r * cell_s), cell_s * 0.9, cell_s * 0.9,
                                     facecolor=color, edgecolor="#94a3b8", linewidth=0.5)
            ax.add_patch(rect)
    ax.text(start_x + 3 * cell_s, start_y - 6.5 * cell_s, "$X_i \\in \\mathbb{R}^{n_i \\times d}$", ha="center", fontsize=8.5, color="#475569")

    # Target Vector y (single strip)
    target_x = start_x + 6.8 * cell_s
    for r in range(6):
        y_val = 1 if r % 2 == 0 else -1
        color = "#fee2e2" if y_val == 1 else "#dcfce7"
        rect = patches.Rectangle((target_x, start_y - r * cell_s), cell_s * 0.9, cell_s * 0.9,
                                 facecolor=color, edgecolor="#e2e8f0", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(target_x + cell_s * 0.45, start_y - r * cell_s + cell_s * 0.45, "+1" if y_val == 1 else "-1",
                ha="center", va="center", fontsize=6, fontweight="bold", color="#991b1b" if y_val == 1 else "#166534")
    ax.text(target_x + cell_s * 0.45, start_y - 6.5 * cell_s, "$y_i$", ha="center", fontsize=8.5, color="#475569")

    # Down Arrow to Statistics
    ax.annotate("", xy=(17.5, 56.5), xytext=(17.5, 59.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748b"))
    ax.text(18.2, 58, "Norm Clip: $\\|z_{it}\\|_2 \\leq C_z$", fontsize=8, color="#475569")

    # 2. Covariance Matrix Heatmaps (Before and After DP)
    ax.text(17.5, 54.5, "Second-Order Empirical Matrix $H_i$", ha="center", fontsize=9.5, fontweight="bold", color="#334155")
    
    # Clean Empirical Heatmap (5x5)
    hm1_x, hm1_y = 6.2, 43.5
    hm_s = 1.35
    cov_matrix = np.array([
        [1.0, 0.72, 0.25, 0.65, 0.81],
        [0.72, 1.0, 0.31, 0.54, 0.76],
        [0.25, 0.31, 1.0, 0.18, 0.29],
        [0.65, 0.54, 0.18, 1.0, 0.62],
        [0.81, 0.76, 0.29, 0.62, 1.0]
    ])
    for r in range(5):
        for c in range(5):
            intensity = cov_matrix[r, c]
            col = plt.cm.Oranges(0.2 + 0.75 * intensity)
            rect = patches.Rectangle((hm1_x + c * hm_s, hm1_y - r * hm_s), hm_s * 0.92, hm_s * 0.92,
                                     facecolor=col, edgecolor="#cbd5e1", linewidth=0.5)
            ax.add_patch(rect)
    ax.text(hm1_x + 2.5 * hm_s, hm1_y - 5.5 * hm_s, "Clean $H_i = \\frac{1}{n_i} Z_i^\\top Z_i$", ha="center", fontsize=8.2, color="#9a3412")

    # Perturbation Arrow
    ax.annotate("", xy=(18.8, 40.5), xytext=(14.8, 40.5),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#dc2626"))
    ax.text(16.8, 42.2, "$+ E_i$", ha="center", fontsize=8.5, fontweight="bold", color="#dc2626")
    ax.text(16.8, 38.8, "$\\mathcal{N}(0, \\sigma_i^2)$", ha="center", fontsize=7.5, color="#dc2626")

    # Privatized Heatmap (5x5)
    hm2_x, hm2_y = 20.0, 43.5
    noisy_matrix = cov_matrix + np.random.normal(0, 0.15, (5, 5))
    noisy_matrix = (noisy_matrix + noisy_matrix.T) / 2
    for r in range(5):
        for c in range(5):
            intensity = np.clip(noisy_matrix[r, c], 0, 1.2) / 1.2
            col = plt.cm.Greens(0.25 + 0.70 * intensity)
            rect = patches.Rectangle((hm2_x + c * hm_s, hm2_y - r * hm_s), hm_s * 0.92, hm_s * 0.92,
                                     facecolor=col, edgecolor="#cbd5e1", linewidth=0.5)
            ax.add_patch(rect)
    ax.text(hm2_x + 2.5 * hm_s, hm2_y - 5.5 * hm_s, "Private $\\widetilde{H}_i = H_i + E_i$", ha="center", fontsize=8.2, color="#166534")

    # 3. Privacy Guarantee Callout Box
    dp_box = patches.FancyBboxPatch((5.5, 9.5), 24, 18.5, boxstyle="round,pad=0.8",
                                    facecolor="#fef2f2", edgecolor="#ef4444", linewidth=1.2)
    ax.add_patch(dp_box)
    ax.text(17.5, 24.5, "Local $(\\epsilon_i, \\delta_i)$-DP Guarantee", ha="center", fontsize=9.5, fontweight="bold", color="#991b1b")
    ax.text(17.5, 19.5, "Global Sensitivity Bound:\n$\\Delta_{2, i} \\leq \\frac{2 C_z^2}{n_i}, \\quad \\sigma_i = \\frac{\\Delta_{2, i} \\sqrt{2\\log(1.25/\\delta)}}{\\epsilon}$",
            ha="center", fontsize=8.5, color="#7f1d1d")
    ax.text(17.5, 13.0, "• One-Shot Release: Zero Iteration Drift\n• Single Uplink: $36$ KB Payload (1 Round)",
            ha="center", fontsize=8.2, color="#991b1b")

    # -------------------------------------------------------------
    # INTER-STAGE ARROW 1 -> 2
    # -------------------------------------------------------------
    ax.annotate("", xy=(34.5, 48), xytext=(32, 48),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#3b82f6", mutation_scale=15))
    ax.text(33.25, 52, "One-Shot\nUpload $\\widetilde{H}_i$", ha="center", fontsize=8.5, fontweight="bold", color="#1d4ed8")

    # -------------------------------------------------------------
    # PANEL B: Stage 2 - Central Collaborative Subspace Pooling (M2)
    # -------------------------------------------------------------
    card_b = patches.FancyBboxPatch((35, 7), 28, 81, boxstyle="round,pad=1.2",
                                    facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5)
    ax.add_patch(card_b)

    hdr_b = patches.FancyBboxPatch((35, 83), 28, 5, boxstyle="round,pad=0.5",
                                   facecolor="#dbeafe", edgecolor="#93c5fd", linewidth=1.2)
    ax.add_patch(hdr_b)
    ax.text(37, 85.5, "b", fontsize=14, fontweight="bold", color="#1e3a8a")
    ax.text(40, 85.5, "Stage 2: Collaborative Pooling", fontsize=11.5, fontweight="bold", color="#1e3a8a")
    ax.text(60, 85.5, "Coordinator $\\mathcal{S}$", fontsize=9, style="italic", color="#3b82f6", ha="right")

    # Core Contribution 2 Badge
    m2_badge = patches.FancyBboxPatch((37, 77.5), 24, 4.5, boxstyle="round,pad=0.4",
                                      facecolor="#00A087", edgecolor="#00A087", linewidth=1.0)
    ax.add_patch(m2_badge)
    ax.text(49, 79.7, "CORE CONTRIBUTION 2 (M2)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#ffffff")

    # Multi-client Matrix Stack Visual
    ax.text(49, 74.5, "Weighted Subspace Aggregation", ha="center", fontsize=9.5, fontweight="bold", color="#1e293b")

    # Stacked 3D-effect layers representing clients C1..CM
    stack_x, stack_y = 40.5, 63.5
    for idx, offset in enumerate([(0, 0), (1.5, 1.2), (3.0, 2.4)]):
        ox, oy = offset
        rect = patches.Rectangle((stack_x + ox, stack_y + oy), 11, 6.5,
                                 facecolor="#ecfdf5" if idx == 2 else "#f1f5f9",
                                 edgecolor="#059669" if idx == 2 else "#94a3b8",
                                 linewidth=1.0, alpha=0.9)
        ax.add_patch(rect)
        label = r"$\widetilde{H}_" + str(idx+1) + r"$ (Client " + str(idx+1) + ")" if idx < 2 else r"$\widetilde{H}_M$ (Client $M$)"
        ax.text(stack_x + ox + 5.5, stack_y + oy + 3.2, label, ha="center", va="center", fontsize=7.5, color="#065f46" if idx == 2 else "#475569")

    # Down Arrow to Aggregation Formula
    ax.annotate("", xy=(49, 56.5), xytext=(49, 61.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))

    # Aggregated Matrix Box with LLN Proof
    pool_box = patches.FancyBboxPatch((37, 33), 24, 21.5, boxstyle="round,pad=0.8",
                                      facecolor="#ecfdf5", edgecolor="#10b981", linewidth=1.4)
    ax.add_patch(pool_box)
    ax.text(49, 50.5, "Global Pooled Matrix $\\widetilde{H}$", ha="center", fontsize=10, fontweight="bold", color="#065f46")
    ax.text(49, 45.5, "$\\widetilde{H} = \\sum_{i=1}^M w_i \\widetilde{H}_i = [\\widetilde{A}, \\widetilde{q}; \\widetilde{q}^\\top, \\widetilde{c}]$",
            ha="center", fontsize=9.5, color="#047857")
    
    # 1/M Variance Reduction Highlight
    lln_badge = patches.FancyBboxPatch((38.5, 35.0), 21, 7.5, boxstyle="round,pad=0.5",
                                       facecolor="#059669", edgecolor="#047857", linewidth=1.0)
    ax.add_patch(lln_badge)
    ax.text(49, 39.8, "1/M Noise Cancellation Bound:", ha="center", fontsize=8.5, fontweight="bold", color="#ffffff")
    ax.text(49, 36.8, "$\\mathrm{Var}(\\widetilde{H}_{jk}) = \\frac{1}{M} \\sigma_0^2$ (75% Noise Reduction!)",
            ha="center", fontsize=8.5, fontweight="bold", color="#ffffff")

    # Down Arrow to PSD Ridge Projection
    ax.annotate("", xy=(49, 26), xytext=(49, 31),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748b"))

    # PSD Projection Box
    psd_box = patches.FancyBboxPatch((37, 9.5), 24, 15, boxstyle="round,pad=0.8",
                                     facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.2)
    ax.add_patch(psd_box)
    ax.text(49, 21.5, "PSD Cone & Ridge Projection", ha="center", fontsize=9.5, fontweight="bold", color="#334155")
    ax.text(49, 17.0, "$\\Pi_{\\geq 0}(\\widetilde{A}) = V \\max(\\Lambda, \\tau I) V^\\top$\n$A \\leftarrow \\Pi_{\\geq 0}(\\widetilde{A}) + \\rho I$",
            ha="center", fontsize=8.8, color="#475569")
    ax.text(49, 12.0, "Guarantees Strict Invertibility & Stability", ha="center", fontsize=7.8, style="italic", color="#64748b")

    # -------------------------------------------------------------
    # INTER-STAGE ARROW 2 -> 3
    # -------------------------------------------------------------
    ax.annotate("", xy=(66, 48), xytext=(63.5, 48),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#E64B35", mutation_scale=15))
    ax.text(64.75, 52, "PSD Matrix\n$[A, q]$", ha="center", fontsize=8.5, fontweight="bold", color="#E64B35")

    # -------------------------------------------------------------
    # PANEL C: Stage 3 - Conditional Orthogonal Variable Screening (M1)
    # -------------------------------------------------------------
    card_c = patches.FancyBboxPatch((66.5, 7), 29.5, 81, boxstyle="round,pad=1.2",
                                    facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5)
    ax.add_patch(card_c)

    hdr_c = patches.FancyBboxPatch((66.5, 83), 29.5, 5, boxstyle="round,pad=0.5",
                                   facecolor="#fee2e2", edgecolor="#fca5a5", linewidth=1.2)
    ax.add_patch(hdr_c)
    ax.text(68.5, 85.5, "c", fontsize=14, fontweight="bold", color="#991b1b")
    ax.text(71.5, 85.5, "Stage 3: Orthogonal Screening", fontsize=11.5, fontweight="bold", color="#991b1b")
    ax.text(93.5, 85.5, "Core M1", fontsize=9, style="italic", color="#ef4444", ha="right")

    # Core Contribution 1 Badge
    m1_badge = patches.FancyBboxPatch((69, 77.5), 24.5, 4.5, boxstyle="round,pad=0.4",
                                      facecolor="#E64B35", edgecolor="#E64B35", linewidth=1.0)
    ax.add_patch(m1_badge)
    ax.text(81.25, 79.7, "CORE CONTRIBUTION 1 (M1)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#ffffff")

    # Visual Schur Projection Formula & Mechanics
    schur_box = patches.FancyBboxPatch((68, 51.5), 26.5, 24.5, boxstyle="round,pad=0.8",
                                       facecolor="#fff1f2", edgecolor="#f43f5e", linewidth=1.4)
    ax.add_patch(schur_box)
    ax.text(81.25, 72.5, "Blockwise Schur Complement Projection", ha="center", fontsize=9.5, fontweight="bold", color="#881337")
    ax.text(81.25, 65.5, "$\\Delta F_j(S) = \\frac{\\left(r_j^{(S)}\\right)^2}{t_j^{(S)}} = \\frac{\\left(q_j - A_{j, S}(A_{SS}+\\rho I)^{-1} q_S\\right)^2}{A_{jj} - A_{j, S}(A_{SS}+\\rho I)^{-1} A_{S, j}}$",
            ha="center", fontsize=8.8, color="#9f1239")
    ax.text(81.25, 57.5, "• Numerator $(r_j^{(S)})^2$: Pure conditional correlation\n• Denominator $t_j^{(S)}$: Residual feature variance\n• Zero Multi-Collinearity: Strips redundant features",
            ha="center", fontsize=8.0, color="#be123c")
    ax.text(81.25, 53.0, "$\\mathbf{\\Delta \\epsilon = 0}$ by DP Post-Processing Invariance",
            ha="center", fontsize=8.2, fontweight="bold", color="#e11d48")

    # Down Arrow to Selection
    ax.annotate("", xy=(81.25, 45.5), xytext=(81.25, 50.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#be123c"))

    # Selected Subset Visual Display
    select_box = patches.FancyBboxPatch((68, 30.0), 26.5, 14.5, boxstyle="round,pad=0.8",
                                        facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.2)
    ax.add_patch(select_box)
    ax.text(81.25, 41.5, "Greedy Iterative Feature Subset $S^*$", ha="center", fontsize=9.5, fontweight="bold", color="#1e293b")
    
    # Feature Circles with selected indices
    circle_y = 35.5
    feat_x = [72.5, 76.5, 81.25, 86.0, 90.0]
    feat_labels = ["$f_{12}$", "$f_{37}$", "$f_4$", "$f_{51}$", "$f_K$"]
    for fx, fl in zip(feat_x, feat_labels):
        circ = patches.Circle((fx, circle_y), 1.5, facecolor="#E64B35", edgecolor="#991b1b", linewidth=1.0)
        ax.add_patch(circ)
        ax.text(fx, circle_y, fl, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#ffffff")
    ax.text(81.25, 31.8, "$S^* = \\arg\\max_{j \\notin S} \\Delta F_j(S), \\quad |S^*| = K \\ll d$", ha="center", fontsize=8.5, color="#475569")

    # Down Arrow to Deployment
    ax.annotate("", xy=(81.25, 24.5), xytext=(81.25, 29.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))

    # Stage 4: Downstream Edge IDS Box
    deploy_box = patches.FancyBboxPatch((68, 9.5), 26.5, 14.5, boxstyle="round,pad=0.8",
                                        facecolor="#f0fdf4", edgecolor="#22c55e", linewidth=1.4)
    ax.add_patch(deploy_box)
    ax.text(81.25, 20.5, "d. Edge Deployment & Real-Time IDS", ha="center", fontsize=9.5, fontweight="bold", color="#14532d")
    ax.text(81.25, 15.5, "Single-Client Latency: 2.43 ~ 3.38 ms", ha="center", fontsize=9.2, fontweight="bold", color="#15803d")
    ax.text(81.25, 11.8, "• 1,818x ~ 2,197x Speedup vs. DPKendall (7.4s)\n• Sub-10ms Line-Rate Industrial SCADA Inference",
            ha="center", fontsize=7.8, color="#166534")

    # Save outputs
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUTPUT_DIR / "fig1_overall_framework.pdf", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUTPUT_DIR / "fig1_overall_framework.png", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(ARTIFACT_DIR / "fig1_overall_framework.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("Nature-style framework generated successfully!")

if __name__ == "__main__":
    draw_nature_framework()
