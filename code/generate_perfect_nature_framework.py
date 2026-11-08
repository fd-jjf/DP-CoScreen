"""Generate a publication-grade Nature-style Framework Diagram for DP-CoScreen.

Completely redesigns and polishes the figure:
1. Replaces all garbled/corrupted text with crisp, mathematically accurate formulas.
2. Fixes all typos:
   - "Schur Complement Projection" (NOT "Secure" or "Solver")
   - "CORE CONTRIBUTION 2 (M2)" (NOT "ORCE ... \Omega")
   - "Local (epsilon, delta)-DP Guarantee" (NOT "Guarnatee")
   - "1/M Noise Cancellation Bound" (NOT "1M ... \tau^2 + p^2")
   - Exact mathematical Schur projection: \Delta F_j(S) = (r_j^{(S)})^2 / t_j^{(S)}
3. Visual iconography:
   - Raw data matrix grid with feature columns and target strip vector (+1/-1)
   - Covariance heatmaps (warm orange clean matrix -> + Gaussian noise -> emerald private matrix)
   - 3D-stacked client matrices for collaborative pooling with 1/M variance cancellation badge
   - Blockwise Schur complement de-redundancy mechanism
   - Feature circle array S* = {f_1, f_2, ..., f_K}
   - Real-time line-rate deployment badge (< 3.5 ms, 2,000x speedup)
4. Saves high-res PDF and PNG directly to manuscript_elsevier/figures/ and artifact directory.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("manuscript_elsevier/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("C:/Users/25358/.gemini/antigravity/brain/77101361-60c7-4fc0-a393-dc926203ec98")

# High quality Nature-standard rendering
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "figure.dpi": 300,
})

def draw_polished_nature_framework():
    fig = plt.figure(figsize=(16.8, 8.2), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.patch.set_facecolor("#ffffff")

    # =========================================================================
    # 0. TOP BANNER
    # =========================================================================
    title_box = patches.FancyBboxPatch((18, 92.2), 64, 6.2, boxstyle="round,pad=0.6",
                                       facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2)
    ax.add_patch(title_box)
    ax.text(50, 96.0, "DP-CoScreen: Differentially Private Collaborative Orthogonal Screening Framework",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color="#0f172a")
    ax.text(50, 93.6, "Non-IID Edge Telemetry  •  One-Shot Gaussian Release  •  1/M Subspace Noise Cancellation  •  Schur Complement Projection",
            ha="center", va="center", fontsize=9.8, style="italic", color="#475569")

    # =========================================================================
    # 1. COLUMN 1: Stage 1 - Edge Gateways (Clients C_1 ... C_M)
    # =========================================================================
    col1_bg = patches.FancyBboxPatch((2.5, 4.5), 29.5, 85.5, boxstyle="round,pad=1.0",
                                     facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5)
    ax.add_patch(col1_bg)

    # Header Card 1
    hdr1 = patches.FancyBboxPatch((2.5, 84.5), 29.5, 5.5, boxstyle="round,pad=0.4",
                                  facecolor="#dbeafe", edgecolor="#93c5fd", linewidth=1.2)
    ax.add_patch(hdr1)
    # Tag 'a'
    tag_a = patches.Circle((5.2, 87.2), 1.5, facecolor="#2563eb", edgecolor="#1d4ed8", linewidth=1.0)
    ax.add_patch(tag_a)
    ax.text(5.2, 87.2, "a", ha="center", va="center", fontsize=11, fontweight="bold", color="#ffffff")
    ax.text(8.0, 87.2, "Stage 1: Edge Gateways", va="center", fontsize=12, fontweight="bold", color="#1e3a8a")
    ax.text(30.5, 87.2, "Clients $C_1 \\dots C_M$", va="center", ha="right", fontsize=9.0, style="italic", color="#3b82f6")

    # --- Box 1.1: Raw Telemetry Stream ---
    b1_1 = patches.FancyBboxPatch((4.0, 68.0), 26.5, 15.0, boxstyle="round,pad=0.6",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.1)
    ax.add_patch(b1_1)
    ax.text(17.25, 80.8, "Raw Telemetry Stream $D_i = \\{(x_{it}, y_{it})\\}$", ha="center", fontsize=9.8, fontweight="bold", color="#1e293b")

    # Data matrix grid: 6 rows x 6 cols features + 1 col label
    gx0, gy0 = 6.2, 77.0
    cs = 1.05
    np.random.seed(42)
    for r in range(5):
        for c in range(6):
            cell_col = "#f1f5f9" if (r + c) % 3 != 0 else "#cbd5e1"
            rect = patches.Rectangle((gx0 + c * cs, gy0 - r * cs), cs * 0.9, cs * 0.9,
                                     facecolor=cell_col, edgecolor="#94a3b8", linewidth=0.5)
            ax.add_patch(rect)
    ax.text(gx0 + 3.0 * cs, gy0 - 5.5 * cs, "$X_i \\in \\mathbb{R}^{n_i \\times d}$ (Flow Features)", ha="center", fontsize=8.0, color="#475569")

    # Target vector strip y
    ty0 = gx0 + 6.6 * cs
    for r in range(5):
        y_val = 1 if r % 2 == 0 else -1
        c_bg = "#fee2e2" if y_val == 1 else "#dcfce7"
        rect = patches.Rectangle((ty0, gy0 - r * cs), cs * 0.9, cs * 0.9,
                                 facecolor=c_bg, edgecolor="#94a3b8", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(ty0 + cs * 0.45, gy0 - r * cs + cs * 0.45, "+1" if y_val == 1 else "-1",
                ha="center", va="center", fontsize=7.0, fontweight="bold", color="#991b1b" if y_val == 1 else "#166534")
    ax.text(ty0 + cs * 0.45, gy0 - 5.5 * cs, "$y_i \\in \\{\\pm 1\\}$", ha="center", fontsize=8.0, color="#475569")

    # Norm clipping badge
    clip_badge = patches.FancyBboxPatch((18.0, 71.5), 11.5, 5.0, boxstyle="round,pad=0.4",
                                        facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.8)
    ax.add_patch(clip_badge)
    ax.text(23.75, 74.8, "Sample-Level Clipping:", ha="center", fontsize=7.5, fontweight="bold", color="#334155")
    ax.text(23.75, 72.7, "$\\|z_{it}\\|_2 \\leq C_z$", ha="center", fontsize=8.5, color="#1e293b")

    # Arrow Down
    ax.annotate("", xy=(17.25, 63.8), xytext=(17.25, 67.5),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#475569"))

    # --- Box 1.2: Second-Order Sufficient Statistics ---
    b1_2 = patches.FancyBboxPatch((4.0, 36.5), 26.5, 26.5, boxstyle="round,pad=0.6",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.1)
    ax.add_patch(b1_2)
    ax.text(17.25, 60.8, "Second-Order Empirical Matrix $H_i$", ha="center", fontsize=9.8, fontweight="bold", color="#1e293b")

    # Covariance heatmaps: Clean H_i (orange) + Noise -> Private H_tilde_i (green)
    hm_s = 1.25
    hm1_x, hm1_y = 5.2, 53.5
    cov_matrix = np.array([
        [1.0, 0.75, 0.20, 0.65, 0.85],
        [0.75, 1.0, 0.35, 0.50, 0.70],
        [0.20, 0.35, 1.0, 0.15, 0.30],
        [0.65, 0.50, 0.15, 1.0, 0.60],
        [0.85, 0.70, 0.30, 0.60, 1.0]
    ])
    for r in range(5):
        for c in range(5):
            intensity = cov_matrix[r, c]
            col = plt.cm.Oranges(0.20 + 0.75 * intensity)
            rect = patches.Rectangle((hm1_x + c * hm_s, hm1_y - r * hm_s), hm_s * 0.92, hm_s * 0.92,
                                     facecolor=col, edgecolor="#fed7aa", linewidth=0.5)
            ax.add_patch(rect)
    ax.text(hm1_x + 2.5 * hm_s, hm1_y - 5.5 * hm_s, "Clean $H_i = \\frac{1}{n_i} Z_i^\\top Z_i$", ha="center", fontsize=8.0, color="#9a3412")

    # Plus & Noise Arrow
    ax.annotate("", xy=(18.4, 50.5), xytext=(15.2, 50.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#dc2626"))
    ax.text(16.8, 52.2, "$+ E_i$", ha="center", fontsize=9.0, fontweight="bold", color="#dc2626")
    ax.text(16.8, 48.8, "$\\mathcal{N}(0, \\sigma_i^2)$", ha="center", fontsize=7.5, color="#dc2626")

    # Private Matrix Heatmap (Green)
    hm2_x, hm2_y = 19.2, 53.5
    noisy_cov = cov_matrix + np.random.normal(0, 0.12, (5, 5))
    noisy_cov = (noisy_cov + noisy_cov.T) / 2
    for r in range(5):
        for c in range(5):
            intensity = np.clip(noisy_cov[r, c], 0, 1.2) / 1.2
            col = plt.cm.Greens(0.25 + 0.70 * intensity)
            rect = patches.Rectangle((hm2_x + c * hm_s, hm2_y - r * hm_s), hm_s * 0.92, hm_s * 0.92,
                                     facecolor=col, edgecolor="#bbf7d0", linewidth=0.5)
            ax.add_patch(rect)
    ax.text(hm2_x + 2.5 * hm_s, hm2_y - 5.5 * hm_s, "Private $\\widetilde{H}_i = H_i + E_i$", ha="center", fontsize=8.0, color="#166534")

    # Formula for H_i block structure
    ax.text(17.25, 38.2, "$H_i = [A_i, q_i; q_i^\\top, 1], \\quad A_i = \\frac{1}{n_i}X_i^\\top X_i, \\; q_i = \\frac{1}{n_i}X_i^\\top y_i$",
            ha="center", fontsize=7.8, color="#334155")

    # Arrow Down
    ax.annotate("", xy=(17.25, 32.2), xytext=(17.25, 35.8),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#475569"))

    # --- Box 1.3: Local DP Guarantee ---
    b1_3 = patches.FancyBboxPatch((4.0, 6.0), 26.5, 25.5, boxstyle="round,pad=0.6",
                                  facecolor="#fef2f2", edgecolor="#ef4444", linewidth=1.2)
    ax.add_patch(b1_3)
    ax.text(17.25, 29.0, "Local $(\\epsilon_i, \\delta_i)$-DP Guarantee", ha="center", fontsize=9.8, fontweight="bold", color="#991b1b")
    ax.text(17.25, 26.2, "(Strict Sample Adjacency Boundedness)", ha="center", fontsize=8.0, style="italic", color="#b91c1c")

    dp_eq_box = patches.FancyBboxPatch((5.5, 14.5), 23.5, 9.8, boxstyle="round,pad=0.4",
                                       facecolor="#ffffff", edgecolor="#fca5a5", linewidth=0.8)
    ax.add_patch(dp_eq_box)
    ax.text(17.25, 21.5, "Global $\\ell_2$ Sensitivity Bound:", ha="center", fontsize=8.0, fontweight="bold", color="#7f1d1d")
    ax.text(17.25, 18.5, "$\\Delta_{2, i} \\leq \\frac{2 C_z^2}{n_i}, \\quad \\sigma_i = \\frac{\\Delta_{2, i} \\sqrt{2\\log(1.25/\\delta_i)}}{\\epsilon_i}$",
            ha="center", fontsize=8.8, color="#991b1b")
    ax.text(17.25, 15.5, "Analytic Gaussian Mechanism", ha="center", fontsize=7.8, style="italic", color="#b91c1c")

    ax.text(17.25, 11.5, "• One-Shot Release: Zero Iteration Drift\n• Single Uplink: $21 \\sim 36$ KB Payload (1 Round)",
            ha="center", fontsize=8.2, fontweight="bold", color="#7f1d1d")
    ax.text(17.25, 7.8, "• Physical Protection: Zero Raw Data Leakage", ha="center", fontsize=8.0, color="#991b1b")

    # =========================================================================
    # INTER-STAGE ARROW 1 -> 2
    # =========================================================================
    ax.annotate("", xy=(35.0, 48.0), xytext=(32.2, 48.0),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#2563eb", mutation_scale=16))
    uplink_box = patches.FancyBboxPatch((32.0, 51.0), 3.5, 7.0, boxstyle="round,pad=0.3",
                                        facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=0.9)
    ax.add_patch(uplink_box)
    ax.text(33.75, 55.5, "One-Shot\nUpload", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1d4ed8")
    ax.text(33.75, 52.5, "$\\widetilde{H}_i$", ha="center", va="center", fontsize=8.5, color="#1e40af")

    # =========================================================================
    # 2. COLUMN 2: Stage 2 - Collaborative Pooling & Subspace Denoising (M2)
    # =========================================================================
    col2_bg = patches.FancyBboxPatch((35.5, 4.5), 29.5, 85.5, boxstyle="round,pad=1.0",
                                     facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5)
    ax.add_patch(col2_bg)

    # Header Card 2
    hdr2 = patches.FancyBboxPatch((35.5, 84.5), 29.5, 5.5, boxstyle="round,pad=0.4",
                                  facecolor="#dcfce7", edgecolor="#86efac", linewidth=1.2)
    ax.add_patch(hdr2)
    tag_b = patches.Circle((38.2, 87.2), 1.5, facecolor="#16a34a", edgecolor="#15803d", linewidth=1.0)
    ax.add_patch(tag_b)
    ax.text(38.2, 87.2, "b", ha="center", va="center", fontsize=11, fontweight="bold", color="#ffffff")
    ax.text(41.0, 87.2, "Stage 2: Sample-Weighted Aggregation", va="center", fontsize=12, fontweight="bold", color="#14532d")
    ax.text(63.5, 87.2, "Coordinator Server $\\mathcal{S}$", va="center", ha="right", fontsize=9.0, style="italic", color="#16a34a")

    # Core Contribution 2 Badge
    m2_badge = patches.FancyBboxPatch((38.0, 78.5), 24.5, 4.8, boxstyle="round,pad=0.4",
                                      facecolor="#00A087", edgecolor="#007765", linewidth=1.2)
    ax.add_patch(m2_badge)
    ax.text(50.25, 80.9, "CORE CONTRIBUTION 2 (M2)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ffffff")

    # --- Box 2.1: Sample-Weighted Aggregation ---
    b2_1 = patches.FancyBboxPatch((37.0, 36.5), 26.5, 40.5, boxstyle="round,pad=0.6",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.1)
    ax.add_patch(b2_1)
    ax.text(50.25, 74.8, "Sample-Weighted Aggregation", ha="center", fontsize=10.0, fontweight="bold", color="#1e293b")

    # 3D Stack of Matrix Cards representing multiple clients
    stack_x, stack_y = 42.0, 64.5
    for idx, (ox, oy, clabel, cbg) in enumerate([
        (0.0, 0.0, "$\\widetilde{H}_1$ (Client 1)", "#f8fafc"),
        (2.0, 1.8, "$\\widetilde{H}_2$ (Client 2)", "#f1f5f9"),
        (4.0, 3.6, "$\\widetilde{H}_M$ (Client $M$)", "#ecfdf5")
    ]):
        st_rect = patches.Rectangle((stack_x + ox, stack_y + oy), 12.5, 6.0,
                                    facecolor=cbg, edgecolor="#059669" if idx == 2 else "#94a3b8",
                                    linewidth=1.0, alpha=0.92)
        ax.add_patch(st_rect)
        ax.text(stack_x + ox + 6.25, stack_y + oy + 3.0, clabel,
                ha="center", va="center", fontsize=8.0, fontweight="bold",
                color="#065f46" if idx == 2 else "#475569")

    # Aggregated Matrix Box
    ax.text(50.25, 59.8, "Aggregated Matrix $\\widetilde{H}$", ha="center", fontsize=9.5, fontweight="bold", color="#065f46")
    ax.text(50.25, 56.5, "$\\widetilde{H} = \\sum_{i=1}^M w_i \\widetilde{H}_i = [\\widetilde{A}, \\widetilde{q}; \\widetilde{q}^\\top, \\widetilde{c}], \\quad w_i = \\frac{n_i}{N}$",
            ha="center", fontsize=8.6, color="#047857")

    # 1/M Perturbation Variance Reduction Callout (Solid Emerald Green)
    lln_box = patches.FancyBboxPatch((38.5, 40.5), 23.5, 13.5, boxstyle="round,pad=0.5",
                                     facecolor="#00A087", edgecolor="#007765", linewidth=1.2)
    ax.add_patch(lln_box)
    ax.text(50.25, 50.8, "1/M Perturbation Variance Reduction:", ha="center", fontsize=9.0, fontweight="bold", color="#ffffff")
    ax.text(50.25, 47.0, "$\\operatorname{Var}(\\bar{E}_{jk}) = \\sum_{i=1}^M w_i^2 \\sigma_i^2 \\approx \\frac{1}{M} \\sigma_0^2$",
            ha="center", fontsize=8.8, fontweight="bold", color="#ffffff")
    ax.text(50.25, 43.0, "75% Variance Reduction ($M=4$)",
            ha="center", fontsize=8.2, fontweight="bold", color="#fef08a")

    ax.text(50.25, 38.0, "Reduces Local Perturbation Variance", ha="center", fontsize=7.8, style="italic", color="#065f46")

    # Arrow Down
    ax.annotate("", xy=(50.25, 29.5), xytext=(50.25, 35.8),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#475569"))

    # --- Box 2.2: PSD Cone & Ridge Projection ---
    b2_2 = patches.FancyBboxPatch((37.0, 6.0), 26.5, 22.8, boxstyle="round,pad=0.6",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.1)
    ax.add_patch(b2_2)
    ax.text(50.25, 25.8, "Positive Semi-Definite (PSD) Projection", ha="center", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax.text(50.25, 23.0, "(Numerical Stability & Spectrum Correction)", ha="center", fontsize=8.0, style="italic", color="#64748b")

    psd_formula_box = patches.FancyBboxPatch((38.5, 12.5), 23.5, 8.8, boxstyle="round,pad=0.4",
                                            facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.8)
    ax.add_patch(psd_formula_box)
    ax.text(50.25, 18.5, "$\\Pi_{\\geq 0}(\\widetilde{A}) = V \\max(\\Lambda, \\tau I) V^\\top$",
            ha="center", fontsize=9.2, color="#0f172a")
    ax.text(50.25, 14.8, "$A \\leftarrow \\Pi_{\\geq 0}(\\widetilde{A}) + \\rho I, \\quad (\\rho = 10^{-3})$",
            ha="center", fontsize=9.0, color="#1e293b")

    ax.text(50.25, 9.8, "• Truncates Noise-Induced Negative Eigenvalues\n• Guarantees Strict Invertibility $(A_{SS} + \\rho I)^{-1}$",
            ha="center", fontsize=8.0, color="#334155")

    # =========================================================================
    # INTER-STAGE ARROW 2 -> 3
    # =========================================================================
    ax.annotate("", xy=(68.0, 48.0), xytext=(65.2, 48.0),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#E64B35", mutation_scale=16))
    psd_tx_box = patches.FancyBboxPatch((65.0, 51.0), 3.5, 7.0, boxstyle="round,pad=0.3",
                                        facecolor="#fef2f2", edgecolor="#fca5a5", linewidth=0.9)
    ax.add_patch(psd_tx_box)
    ax.text(66.75, 55.5, "PSD Matrix", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#b91c1c")
    ax.text(66.75, 52.5, "$[A, q]$", ha="center", va="center", fontsize=8.5, color="#991b1b")

    # =========================================================================
    # 3. COLUMN 3: Stage 3 - Orthogonal Variable Screening & Deployment (M1)
    # =========================================================================
    col3_bg = patches.FancyBboxPatch((68.5, 4.5), 29.0, 85.5, boxstyle="round,pad=1.0",
                                     facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5)
    ax.add_patch(col3_bg)

    # Header Card 3
    hdr3 = patches.FancyBboxPatch((68.5, 84.5), 29.0, 5.5, boxstyle="round,pad=0.4",
                                  facecolor="#fee2e2", edgecolor="#fca5a5", linewidth=1.2)
    ax.add_patch(hdr3)
    tag_c = patches.Circle((71.2, 87.2), 1.5, facecolor="#dc2626", edgecolor="#b91c1c", linewidth=1.0)
    ax.add_patch(tag_c)
    ax.text(71.2, 87.2, "c", ha="center", va="center", fontsize=11, fontweight="bold", color="#ffffff")
    ax.text(74.0, 87.2, "Stage 3: Orthogonal Screening", va="center", fontsize=12, fontweight="bold", color="#991b1b")
    ax.text(96.0, 87.2, "Schur Complement", va="center", ha="right", fontsize=9.0, style="italic", color="#ef4444")

    # Core Contribution 1 Badge
    m1_badge = patches.FancyBboxPatch((70.5, 78.5), 25.0, 4.8, boxstyle="round,pad=0.4",
                                      facecolor="#E64B35", edgecolor="#b91c1c", linewidth=1.2)
    ax.add_patch(m1_badge)
    ax.text(83.0, 80.9, "CORE CONTRIBUTION 1 (M1)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ffffff")

    # --- Box 3.1: Blockwise Schur Complement Projection ---
    b3_1 = patches.FancyBboxPatch((70.0, 46.5), 26.0, 30.5, boxstyle="round,pad=0.6",
                                  facecolor="#fff1f2", edgecolor="#f43f5e", linewidth=1.2)
    ax.add_patch(b3_1)
    ax.text(83.0, 74.5, "Blockwise Schur Complement Projection", ha="center", fontsize=10.0, fontweight="bold", color="#881337")

    # Schur Equation Box
    schur_eq_box = patches.FancyBboxPatch((71.0, 62.5), 24.0, 10.0, boxstyle="round,pad=0.4",
                                          facecolor="#ffffff", edgecolor="#fbcfe8", linewidth=0.8)
    ax.add_patch(schur_eq_box)
    ax.text(83.0, 69.2, "$\\Delta F_j(S) = \\frac{\\left(r_j^{(S)}\\right)^2}{t_j^{(S)}} = \\frac{\\left(q_j - A_{j, S}(A_{SS}+\\rho I)^{-1} q_S\\right)^2}{A_{jj} - A_{j, S}(A_{SS}+\\rho I)^{-1} A_{S, j}}$",
            ha="center", fontsize=8.6, color="#9f1239")
    ax.text(83.0, 64.5, "Analytical Closed-Form Incremental Criterion", ha="center", fontsize=7.6, style="italic", color="#be123c")

    ax.text(83.0, 58.5, "• Numerator $(r_j^{(S)})^2$: Pure conditional correlation\n• Denominator $t_j^{(S)}$: Residual feature variance\n• Zero Multi-Collinearity: Strips redundant features",
            ha="center", fontsize=8.0, color="#881337")

    # Post-Processing Privacy Badge
    pp_badge = patches.FancyBboxPatch((71.5, 48.0), 23.0, 4.2, boxstyle="round,pad=0.3",
                                      facecolor="#ffffff", edgecolor="#e11d48", linewidth=1.0)
    ax.add_patch(pp_badge)
    ax.text(83.0, 50.1, "$\\mathbf{\\Delta \\epsilon = 0}$ by DP Post-Processing Invariance!",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color="#e11d48")

    # Arrow Down
    ax.annotate("", xy=(83.0, 42.2), xytext=(83.0, 45.8),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#475569"))

    # --- Box 3.2: Greedy Selected Feature Subset S* ---
    b3_2 = patches.FancyBboxPatch((70.0, 27.5), 26.0, 14.0, boxstyle="round,pad=0.6",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.1)
    ax.add_patch(b3_2)
    ax.text(83.0, 39.2, "Greedy Iterative Feature Subset $S^*$", ha="center", fontsize=9.5, fontweight="bold", color="#1e293b")

    # Feature Circles with selected indices in Nature Red
    circ_y = 34.0
    feat_x = [73.5, 77.5, 81.5, 85.5, 89.5, 93.0]
    feat_lbls = ["$f_{12}$", "$f_{37}$", "$f_4$", "$f_{51}$", "$f_K$", "..."]
    for fx, fl in zip(feat_x, feat_lbls):
        if fl != "...":
            circ = patches.Circle((fx, circ_y), 1.6, facecolor="#E64B35", edgecolor="#991b1b", linewidth=1.0)
            ax.add_patch(circ)
            ax.text(fx, circ_y, fl, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#ffffff")
        else:
            ax.text(fx, circ_y, fl, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#64748b")
    ax.text(83.0, 29.5, "$j^* = \\arg\\max_{j \\notin S} \\Delta F_j(S), \\quad |S^*| = K \\ll d$", ha="center", fontsize=8.5, color="#475569")

    # Arrow Down
    ax.annotate("", xy=(83.0, 23.2), xytext=(83.0, 26.8),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#059669"))

    # --- Box 3.3: Stage 4 / Deployment - Real-Time Edge IDS ---
    b3_3 = patches.FancyBboxPatch((70.0, 6.0), 26.0, 16.5, boxstyle="round,pad=0.6",
                                  facecolor="#f0fdf4", edgecolor="#22c55e", linewidth=1.3)
    ax.add_patch(b3_3)
    # Tag 'd'
    tag_d = patches.Circle((72.2, 19.5), 1.3, facecolor="#16a34a", edgecolor="#15803d", linewidth=0.8)
    ax.add_patch(tag_d)
    ax.text(72.2, 19.5, "d", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ffffff")
    ax.text(74.5, 19.5, "Edge Deployment & Intrusion Detection", va="center", fontsize=9.8, fontweight="bold", color="#14532d")

    ax.text(83.0, 15.2, "Feature-Selection Runtime: 2.43 ~ 3.38 ms", ha="center", fontsize=9.2, fontweight="bold", color="#15803d")
    ax.text(83.0, 11.8, "• $\\approx 1.8 \\times 10^3\\times$ to $2.2 \\times 10^3\\times$ Ratio vs. DPKendall\n• Evaluated on ToN-IoT & Edge-IIoTset Benchmarks",
            ha="center", fontsize=7.8, color="#166534")
    ax.text(83.0, 7.8, "• High Predictive Utility & Formal DP Guarantee", ha="center", fontsize=7.8, fontweight="bold", color="#15803d")

    # Save
    fig.savefig(OUTPUT_DIR / "fig1_overall_framework.pdf", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUTPUT_DIR / "fig1_overall_framework.png", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(ARTIFACT_DIR / "fig1_overall_framework.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("Polished Nature-style framework diagram generated successfully!")

if __name__ == "__main__":
    draw_polished_nature_framework()
