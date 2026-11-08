from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, subspace_angles
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class SyntheticConfig:
    n_clients: int = 4
    feature_dim: int = 24
    shared_rank: int = 3
    personalized_rank: int = 2
    shared_strength: float = 3.0
    personal_strength: float = 1.4
    personal_signal_scale: float = 1.0
    label_strength: float = 1.2
    cov_noise: float = 0.25
    label_noise: float = 0.5
    clip_norm: float = 4.0
    train_size: int = 600
    test_size: int = 200
    pop_size: int = 5000
    epsilon: float = 1.0
    delta: float = 1e-5
    lambda_s: float = 0.2
    lambda_p: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.4
    selected_k: int = 8
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClientData:
    sigma_x: np.ndarray
    beta: np.ndarray
    x_train_full: np.ndarray
    y_train_full: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_pop: np.ndarray
    y_pop: np.ndarray
    H_emp_full: np.ndarray
    H_pop: np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _clip_rows_l2(x: np.ndarray, clip_norm: float) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    return x * scale


def _augment_and_clip(
    x: np.ndarray, y: np.ndarray, clip_norm: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of rows")
    z = np.column_stack([x, y])
    raw_norms = np.linalg.norm(z, axis=1)
    scales = np.minimum(1.0, clip_norm / np.maximum(raw_norms, 1e-12))
    return z * scales[:, None], raw_norms, scales


def _orthonormal_columns(
    rng: np.random.Generator,
    dim: int,
    rank: int,
    orth_to: np.ndarray | None = None,
) -> np.ndarray:
    if rank <= 0:
        return np.zeros((dim, 0))
    m = rng.normal(size=(dim, rank + max(2, dim)))
    if orth_to is not None and orth_to.size:
        m = m - orth_to @ (orth_to.T @ m)
    q, _ = np.linalg.qr(m)
    return q[:, :rank]


def _sample_symmetric_gaussian(
    rng: np.random.Generator, p: int, sigma: float
) -> np.ndarray:
    coeffs = rng.normal(scale=sigma, size=p * (p + 1) // 2)
    w = np.zeros((p, p))
    idx = 0
    for i in range(p):
        w[i, i] = coeffs[idx]
        idx += 1
    scale = np.sqrt(2.0)
    for i in range(p):
        for j in range(i + 1, p):
            val = coeffs[idx] / scale
            w[i, j] = val
            w[j, i] = val
            idx += 1
    return w


def _structure_from_xy(x: np.ndarray, y: np.ndarray, clip_norm: float) -> np.ndarray:
    z_clipped, _, _ = _augment_and_clip(x, y, clip_norm)
    return _symmetrize((z_clipped.T @ z_clipped) / z_clipped.shape[0])


def _compute_sigma(n: int, clip_norm: float, epsilon: float, delta: float) -> float:
    sensitivity = 2.0 * clip_norm**2 / max(n, 1)
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / max(epsilon, 1e-12)


def _generate_client_params(
    rng: np.random.Generator,
    feature_dim: int,
    shared_basis: np.ndarray,
    shared_strength: float,
    personal_strength: float,
    label_strength: float,
    cov_noise: float,
    heterogeneity_scale: float,
    personal_signal_scale: float,
    shared_rank: int,
    personalized_rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    personal_basis = _orthonormal_columns(
        rng, feature_dim, personalized_rank, orth_to=shared_basis
    )
    if personal_basis.shape[1] < personalized_rank:
        extra = _orthonormal_columns(rng, feature_dim, personalized_rank)
        personal_basis = np.concatenate([personal_basis, extra], axis=1)[:, :personalized_rank]

    shared_cov_eigs = shared_strength * np.linspace(1.0, 0.65, shared_rank)
    personal_cov_eigs = (
        heterogeneity_scale
        * personal_strength
        * np.linspace(1.0, 0.6, personalized_rank)
    )

    shared_beta = shared_basis @ (
        label_strength * np.linspace(1.0, 0.7, shared_rank)
    )
    personal_beta = personal_basis @ (
        heterogeneity_scale
        * personal_signal_scale
        * label_strength
        * np.linspace(0.9, 0.55, personalized_rank)
    )

    sigma_x = (
        shared_basis @ np.diag(shared_cov_eigs) @ shared_basis.T
        + personal_basis @ np.diag(personal_cov_eigs) @ personal_basis.T
        + cov_noise * np.eye(feature_dim)
    )
    beta = shared_beta + personal_beta
    return sigma_x, beta


def _sample_client_dataset(
    rng: np.random.Generator,
    sigma_x: np.ndarray,
    beta: np.ndarray,
    n: int,
    label_noise: float,
    clip_norm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.multivariate_normal(np.zeros(sigma_x.shape[0]), sigma_x, size=n)
    logits = x @ beta + rng.normal(scale=label_noise, size=n)
    y = np.where(logits >= 0, 1.0, -1.0)
    H = _structure_from_xy(x, y, clip_norm)
    return x, y, H


def build_synthetic_world(cfg: SyntheticConfig) -> List[ClientData]:
    base_rng = _rng(cfg.seed)
    feature_dim = cfg.feature_dim
    shared_basis = _orthonormal_columns(base_rng, feature_dim, cfg.shared_rank)

    clients: List[ClientData] = []
    for client_idx in range(cfg.n_clients):
        client_rng = _rng(cfg.seed + 31 * (client_idx + 1))
        heterogeneity_scale = 0.85 + 0.35 * client_rng.random()
        sigma_x, beta = _generate_client_params(
            client_rng,
            feature_dim,
            shared_basis,
            cfg.shared_strength,
            cfg.personal_strength,
            cfg.label_strength,
            cfg.cov_noise,
            heterogeneity_scale,
            cfg.personal_signal_scale,
            cfg.shared_rank,
            cfg.personalized_rank,
        )

        x_train, y_train, _ = _sample_client_dataset(
            client_rng,
            sigma_x,
            beta,
            cfg.train_size,
            cfg.label_noise,
            cfg.clip_norm,
        )
        x_test, y_test, _ = _sample_client_dataset(
            _rng(cfg.seed + 1000 + client_idx),
            sigma_x,
            beta,
            cfg.test_size,
            cfg.label_noise,
            cfg.clip_norm,
        )
        x_pop, y_pop, H_pop = _sample_client_dataset(
            _rng(cfg.seed + 2000 + client_idx),
            sigma_x,
            beta,
            cfg.pop_size,
            cfg.label_noise,
            cfg.clip_norm,
        )
        H_emp_full = _structure_from_xy(x_train, y_train, cfg.clip_norm)
        clients.append(
            ClientData(
                sigma_x=sigma_x,
                beta=beta,
                x_train_full=x_train,
                y_train_full=y_train,
                x_test=x_test,
                y_test=y_test,
                x_pop=x_pop,
                y_pop=y_pop,
                H_emp_full=H_emp_full,
                H_pop=H_pop,
            )
        )
    return clients


def population_operator(
    H_list: Sequence[np.ndarray],
    weights: np.ndarray,
    lambda_s: float,
) -> np.ndarray:
    bar_h = sum(w * h for w, h in zip(weights, H_list))
    K = np.zeros_like(H_list[0])
    for w, h in zip(weights, H_list):
        K += w * (h @ h)
    for w, h in zip(weights, H_list):
        diff = h - bar_h
        K -= lambda_s * w * (diff @ diff)
    return _symmetrize(K)


def dp_shared_operator(
    H_list: Sequence[np.ndarray],
    weights: np.ndarray,
    sigmas: Sequence[float],
    lambda_s: float,
) -> np.ndarray:
    bar_h = sum(w * h for w, h in zip(weights, H_list))
    p = H_list[0].shape[0]
    cp = (p + 1) / 2.0
    K = np.zeros((p, p), dtype=float)
    for w, h, sigma in zip(weights, H_list, sigmas):
        K += w * (h @ h - cp * sigma**2 * np.eye(p))
    for i, (w_i, h_i, sigma_i) in enumerate(zip(weights, H_list, sigmas)):
        tau2 = (1.0 - w_i) ** 2 * sigma_i**2
        for j, (w_j, sigma_j) in enumerate(zip(weights, sigmas)):
            if j == i:
                continue
            tau2 += (w_j**2) * sigma_j**2
        diff = h_i - bar_h
        K -= lambda_s * w_i * (diff @ diff - cp * tau2 * np.eye(p))
    return _symmetrize(K)


def leading_subspace(matrix: np.ndarray, rank: int) -> np.ndarray:
    eigvals, eigvecs = eigh(_symmetrize(matrix))
    idx = np.argsort(eigvals)[::-1][:rank]
    basis = eigvecs[:, idx]
    q, _ = np.linalg.qr(basis)
    return q[:, :rank]


def abs_spectrum_subspace(matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = eigh(_symmetrize(matrix))
    idx = np.argsort(np.abs(eigvals))[::-1][:rank]
    basis = eigvecs[:, idx]
    q, _ = np.linalg.qr(basis)
    return q[:, :rank], eigvals[idx]


def shared_structure_from_subspace(
    subspace: np.ndarray,
    H_list: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    B = np.zeros((subspace.shape[1], subspace.shape[1]), dtype=float)
    for w, h in zip(weights, H_list):
        B += w * (subspace.T @ h @ subspace)
    B = _symmetrize(B)
    return _symmetrize(subspace @ B @ subspace.T)


def residual_and_personal_structure(
    H: np.ndarray,
    shared_subspace: np.ndarray,
    rank: int,
    lambda_p: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = np.eye(H.shape[0]) - shared_subspace @ shared_subspace.T
    residual = _symmetrize(Q @ H @ Q)
    U, eigvals = abs_spectrum_subspace(residual, rank)
    shrunk = np.sign(eigvals) * np.maximum(np.abs(eigvals) - lambda_p, 0.0)
    H_p = _symmetrize(U @ np.diag(shrunk) @ U.T)
    return residual, H_p, U


def conditional_score(
    H_s: np.ndarray,
    H_p: np.ndarray,
    H_struct: np.ndarray,
    selected: Sequence[int],
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    q_s = H_s[:-1, -1]
    q_p = H_p[:-1, -1]
    A = H_struct[:-1, :-1]
    d = A.shape[0]
    selected = list(selected)
    scores = np.full(d, -np.inf, dtype=float)
    for j in range(d):
        if j in selected:
            continue
        red = 0.0
        if selected:
            red = float(np.mean(np.abs(A[j, selected])))
        scores[j] = alpha * abs(q_s[j]) + beta * abs(q_p[j]) - gamma * red
    return scores


def greedy_select(
    H_s: np.ndarray,
    H_p: np.ndarray,
    H_struct: np.ndarray,
    k: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    selected: List[int] = []
    trace: List[Dict[str, Any]] = []
    for t in range(k):
        scores = conditional_score(H_s, H_p, H_struct, selected, alpha, beta, gamma)
        choice = int(np.argmax(scores))
        trace.append(
            {
                "step": t,
                "choice": choice,
                "score": float(scores[choice]),
                "scores": scores.tolist(),
            }
        )
        selected.append(choice)
    return np.asarray(selected, dtype=int), trace


def oracle_feature_set(
    H_pop_list: Sequence[np.ndarray],
    weights: np.ndarray,
    shared_rank: int,
    personal_rank: int,
    k: int,
    alpha: float,
    beta: float,
    gamma: float,
    lambda_s: float,
    lambda_p: float,
) -> Dict[str, Any]:
    K_pop = population_operator(H_pop_list, weights, lambda_s=lambda_s)
    P_pop = leading_subspace(K_pop, shared_rank)
    H_s_pop = shared_structure_from_subspace(P_pop, H_pop_list, weights)
    oracle = []
    residuals = []
    H_p_list = []
    for H in H_pop_list:
        residual, H_p, _ = residual_and_personal_structure(
            H, P_pop, personal_rank, lambda_p=lambda_p
        )
        residuals.append(residual)
        H_p_list.append(H_p)
        selected, trace = greedy_select(H_s_pop, H_p, H, k, alpha, beta, gamma)
        oracle.append({"selected": selected.tolist(), "trace": trace})
    return {
        "K_pop": K_pop,
        "P_pop": P_pop,
        "H_s_pop": H_s_pop,
        "residuals": residuals,
        "H_p_pop_list": H_p_list,
        "oracle": oracle,
    }


def select_random(d: int, k: int, seed: int) -> np.ndarray:
    rng = _rng(seed)
    return rng.choice(d, size=k, replace=False)


def select_local_relevance(H_emp: np.ndarray, k: int) -> np.ndarray:
    scores = np.abs(H_emp[:-1, -1])
    return np.argsort(scores)[::-1][:k]


def select_joint(H_emp: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = _rng(seed)
    A = np.abs(H_emp[:-1, :-1])
    np.fill_diagonal(A, 0.0)
    scores0 = np.abs(H_emp[:-1, -1])
    selected: List[int] = []
    for _ in range(k):
        scores = scores0.copy()
        if selected:
            scores = scores - np.mean(A[selected, :], axis=0)
        scores[selected] = -np.inf
        top = np.flatnonzero(scores == np.max(scores))
        chosen = int(rng.choice(top))
        selected.append(chosen)
    return np.asarray(selected, dtype=int)


def select_dp_kendall(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    epsilon: float,
    seed: int,
) -> np.ndarray:
    rng = _rng(seed)
    n, d = X.shape
    label_corr = np.array([_kendall_tau_abs(X[:, j], y, n) for j in range(d)])
    selected: List[int] = []
    pair_corr = np.zeros((k, d))
    eps_round = max(epsilon / k, 1e-12)
    for t in range(k):
        if t == 0:
            scores = label_corr.copy()
            sensitivity = 1.5
        else:
            scores = label_corr - np.mean(pair_corr[:t, :], axis=0)
            sensitivity = 3.0
        scores[selected] = -np.inf
        beta = 2.0 * sensitivity / eps_round
        noisy_scores = scores + rng.gumbel(0, beta, size=d)
        choice = int(np.argmax(noisy_scores))
        selected.append(choice)
        pair_corr[t, :] = np.array([_kendall_tau_abs(X[:, choice], X[:, j], n) for j in range(d)])
    return np.asarray(selected, dtype=int)


def _kendall_tau_abs(feature: np.ndarray, labels: np.ndarray, n: int) -> float:
    tau, _ = kendalltau(feature, labels, method="asymptotic")
    if np.isnan(tau):
        tau = 0.0
    return abs((n / 2.0) * float(tau))


def select_dpspsr(
    H_tilde_list: Sequence[np.ndarray],
    sigmas: Sequence[float],
    weights: np.ndarray,
    cfg: SyntheticConfig,
) -> Dict[str, Any]:
    K_hat = dp_shared_operator(H_tilde_list, weights, sigmas, cfg.lambda_s)
    P_hat = leading_subspace(K_hat, cfg.shared_rank)
    H_s_hat = shared_structure_from_subspace(P_hat, H_tilde_list, weights)
    selected_sets = []
    residuals = []
    H_p_list = []
    trace_list = []
    for H_tilde in H_tilde_list:
        residual, H_p, _ = residual_and_personal_structure(
            H_tilde, P_hat, cfg.personalized_rank, cfg.lambda_p
        )
        residuals.append(residual)
        H_p_list.append(H_p)
        selected, trace = greedy_select(
            H_s_hat, H_p, H_tilde, cfg.selected_k, cfg.alpha, cfg.beta, cfg.gamma
        )
        selected_sets.append(selected)
        trace_list.append(trace)
    return {
        "K_hat": K_hat,
        "P_hat": P_hat,
        "H_s_hat": H_s_hat,
        "residuals": residuals,
        "H_p_list": H_p_list,
        "selected_sets": selected_sets,
        "traces": trace_list,
    }


def evaluate_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected_idx: Sequence[int],
) -> Dict[str, float]:
    selected_idx = np.asarray(selected_idx, dtype=int)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train[:, selected_idx])
    x_te = scaler.transform(x_test[:, selected_idx])
    y_tr = (y_train > 0).astype(int)
    y_te = (y_test > 0).astype(int)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
    clf.fit(x_tr, y_tr)
    prob = clf.predict_proba(x_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "recall": float(recall_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, prob)),
    }


def jaccard_score(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(map(int, a))
    sb = set(map(int, b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def redundancy_score(H: np.ndarray, selected: Sequence[int]) -> float:
    if len(selected) <= 1:
        return 0.0
    A = np.abs(H[:-1, :-1])
    idx = list(map(int, selected))
    sub = A[np.ix_(idx, idx)]
    return float((np.sum(sub) - np.trace(sub)) / max(len(idx) * (len(idx) - 1), 1))


def rank_flip_count(est: Sequence[int], oracle: Sequence[int]) -> int:
    est = list(map(int, est))
    oracle = list(map(int, oracle))
    k = min(len(est), len(oracle))
    return sum(int(est[i] != oracle[i]) for i in range(k))


def subspace_error(p_hat: np.ndarray, p_true: np.ndarray) -> float:
    if p_hat.size == 0 or p_true.size == 0:
        return 0.0
    angles = subspace_angles(p_hat, p_true)
    return float(np.sin(np.max(angles)))


def operator_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, 2))


def _plot_line(
    xs: Sequence[float],
    ys_dict: Dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(6.2, 4.1))
    for name, ys in ys_dict.items():
        plt.plot(xs, ys, marker="o", linewidth=2, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_multi_panel(
    xs: Sequence[float],
    panels: Sequence[Tuple[str, Dict[str, Sequence[float]], str]],
    xlabel: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, ys_dict, ylabel) in zip(axes, panels):
        for name, ys in ys_dict.items():
            ax.plot(xs, ys, marker="o", linewidth=2, label=name)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


