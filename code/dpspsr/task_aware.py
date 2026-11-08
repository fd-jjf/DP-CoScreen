"""Opt-in task-aware recovery; the legacy spectral selector is unchanged.

Inputs are ONE private release of the second moment of [x, y, 1].
All recovery, gating, centering and selection are postprocessing. The gate is
an empirical noise-risk heuristic, not a theorem about classification utility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class TaskAwareConfig:
    k: int = 10
    ridge_floor: float = 1e-3
    noise_ridge_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if (isinstance(self.k, (bool, np.bool_)) or not isinstance(self.k, (int, np.integer))
                or self.k < 1 or not np.isfinite(self.ridge_floor) or self.ridge_floor <= 0
                or not np.isfinite(self.noise_ridge_multiplier) or self.noise_ridge_multiplier < 0):
            raise ValueError("Invalid selection budget or ridge parameters")

    def manifest(self) -> dict:
        return asdict(self)


def intercept_structure(x: np.ndarray, y: np.ndarray, clip_norm: float) -> np.ndarray:
    """Bound [clipped x, +/-1 label, 1] without data-fitted preprocessing.

    The intercept is exactly one, so centering needs no noisy denominator.
    Replace-one Frobenius sensitivity remains at most 2*C**2/public_n.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).reshape(-1)
    if (x.ndim != 2 or len(x) != len(y) or not len(y)
            or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()
            or not np.isfinite(clip_norm) or clip_norm <= np.sqrt(2)):
        raise ValueError("Need finite features, 0/1 labels and C > sqrt(2)")
    x_radius = np.sqrt(clip_norm**2 - 2)
    scales = np.minimum(1.0, x_radius / np.maximum(np.linalg.norm(x, axis=1), 1e-12))
    z = np.column_stack([x * scales[:, None], 2 * y - 1, np.ones(len(y))])
    return z.T @ z / len(z)


def center_release(h: np.ndarray, sigma: float, centered: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return a centered joint feature/label matrix and plug-in q-noise variances.

    All used moments are off-diagonal entries of the symmetric release and
    have variance v=sigma**2/2, including means paired with the fixed
    intercept coordinate. For fixed true means,
    Var(noisy E[xy] - noisy E[x]*noisy E[y]) is
    v*(1+mu_x**2+mu_y**2)+v**2. Replacing squared means by nonnegative
    debiased estimates is a heuristic; no optimal gate is claimed.
    """
    h = np.asarray(h, dtype=float)
    v = sigma**2 / 2
    moment = (h[:-1, :-1] + h[:-1, :-1].T) / 2
    if not centered:
        return moment, np.full(len(moment) - 1, v)
    means = h[:-1, -1]
    covariance = moment - np.outer(means, means)
    squared_means = np.maximum(means**2 - v, 0)
    q_variance = v * (1 + squared_means[:-1] + squared_means[-1]) + v**2
    return covariance, q_variance


def psd_projection(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((a + a.T) / 2)
    return (vectors * np.maximum(values, 0)) @ vectors.T


def conditional_select(joint: np.ndarray, k: int, ridge: float,
                       conditional: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Greedy reduction in a ridge quadratic objective, not raw covariance mRMR."""
    joint = np.asarray(joint, dtype=float)
    if (joint.ndim != 2 or joint.shape[0] != joint.shape[1]
            or not np.isfinite(joint).all() or not 1 <= k < len(joint)
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("Invalid joint moment, budget or ridge")
    joint = psd_projection(joint)
    a, q = joint[:-1, :-1], joint[:-1, -1]
    return conditional_select_blocks(a, q, k, ridge, conditional)


def conditional_select_blocks(a: np.ndarray, q: np.ndarray, k: int, ridge: float,
                              conditional: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Select using a PSD feature covariance and a separate signed task vector.

    Only A is projected: a noisy label variance cannot alter the recovered q.
    The quadratic surrogate is valid without claiming a joint covariance model.
    """
    a, q = np.asarray(a, dtype=float), np.asarray(q, dtype=float)
    if (a.ndim != 2 or a.shape != (q.size, q.size) or q.ndim != 1
            or not np.isfinite(a).all() or not np.isfinite(q).all()
            or isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 1 <= k <= len(q) or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("Invalid covariance, task vector, budget or ridge")
    a = psd_projection(a)
    d = len(q)
    selected: list[int] = []
    trace = []
    for step in range(k):
        residual_q = q.copy()
        residual_var = np.diag(a).copy() + ridge
        if selected and conditional:
            cross = a[:, selected]
            ass = a[np.ix_(selected, selected)] + ridge * np.eye(len(selected))
            residual_q -= cross @ np.linalg.solve(ass, q[selected])
            residual_var -= np.sum(cross * np.linalg.solve(ass, cross.T).T, axis=1)
        scores = residual_q**2 / np.maximum(residual_var, ridge)
        scores[selected] = -np.inf
        choice = int(np.argmax(scores))
        selected.append(choice)
        trace.append(dict(step=step, choice=choice, gain=float(scores[choice])))
    return np.asarray(selected, dtype=int), trace


def soft_threshold(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Coordinatewise soft thresholding with fixed nonnegative thresholds."""
    values, thresholds = np.asarray(values, dtype=float), np.asarray(thresholds, dtype=float)
    if (values.shape != thresholds.shape or values.ndim != 1
            or not np.isfinite(values).all() or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0)):
        raise ValueError("Invalid values or thresholds")
    return np.sign(values) * np.maximum(np.abs(values) - thresholds, 0)


def quadratic_objective(a: np.ndarray, q: np.ndarray, subset: np.ndarray,
                        ridge: float) -> float:
    """Ridge quadratic selection surrogate for a fixed subset."""
    subset = np.asarray(subset, dtype=int)
    if (a.ndim != 2 or a.shape != (len(q), len(q)) or subset.ndim != 1
            or len(np.unique(subset)) != len(subset) or np.any(subset < 0)
            or np.any(subset >= len(q)) or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("Invalid quadratic objective inputs")
    if not len(subset):
        return 0.0
    ass = a[np.ix_(subset, subset)] + ridge * np.eye(len(subset))
    return float(q[subset] @ np.linalg.solve(ass, q[subset]))


def anchored_swaps(a: np.ndarray, q: np.ndarray, shared: np.ndarray, ridge: float,
                   max_replacements: int, enabled: bool) -> tuple[np.ndarray, list[dict]]:
    """Improve a shared subset with a public cap on personalized replacements."""
    current = np.asarray(shared, dtype=int).copy()
    if (isinstance(max_replacements, (bool, np.bool_))
            or not isinstance(max_replacements, (int, np.integer))
            or not 0 <= max_replacements <= len(current)):
        raise ValueError("Invalid personalized replacement budget")
    if not enabled or max_replacements == 0:
        return current, []
    trace = []
    value = quadratic_objective(a, q, current, ridge)
    for step in range(max_replacements):
        best = None
        chosen = set(current.tolist())
        for incoming in range(len(q)):
            if incoming in chosen:
                continue
            for position, outgoing in enumerate(current):
                candidate = current.copy()
                candidate[position] = incoming
                candidate_value = quadratic_objective(a, q, candidate, ridge)
                gain = candidate_value - value
                if best is None or gain > best[0]:
                    best = (gain, int(incoming), int(outgoing), candidate, candidate_value)
        if best is None or best[0] <= 1e-15:
            break
        gain, incoming, outgoing, current, value = best
        trace.append(dict(step=step, incoming=incoming, outgoing=outgoing,
                          gain=float(gain)))
    return current, trace


def recover_task_aware(h_list: list[np.ndarray], sigmas: np.ndarray, weights: np.ndarray,
                       config: TaskAwareConfig, *, personal_weight: float | None = None,
                       centered: bool = True, conditional: bool = True,
                       fixed_ridge: float | None = None) -> dict:
    """Keep signed task-relevance cross blocks instead of projecting them away."""
    weights = np.asarray(weights, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if (not h_list or weights.shape != (len(h_list),) or sigmas.shape != weights.shape
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1) or not np.isfinite(sigmas).all()
            or np.any(sigmas < 0) or any(h.shape != h_list[0].shape for h in h_list)
            or any(h.ndim != 2 or h.shape[0] != h.shape[1] or len(h) < 3
                   or not np.isfinite(h).all() for h in h_list)
            or (personal_weight is not None and not 0 <= personal_weight <= 1)
            or (fixed_ridge is not None and (not np.isfinite(fixed_ridge) or fixed_ridge <= 0))):
        raise ValueError("Invalid releases, public weights, sigmas or personal weight")
    moments, variances = zip(*(center_release(h, s, centered) for h, s in zip(h_list, sigmas)))
    pooled = sum(w * c for w, c in zip(weights, moments))
    pooled_var = sum(w**2 * v for w, v in zip(weights, variances))
    selected_sets, recovered, diagnostics, traces = [], [], [], []
    for i, (c, v) in enumerate(zip(moments, variances)):
        contrast = c[:-1, -1] - pooled[:-1, -1]
        # Cov(local, pooled) is w_i * Var(local), since the pool includes i.
        contrast_var = np.maximum(v + pooled_var - 2 * weights[i] * v, 0)
        observed_energy = float(contrast @ contrast)
        noise_energy = float(contrast_var.sum())
        gate = (float(np.clip(1 - noise_energy / observed_energy, 0, 1))
                if observed_energy > 0 else 0.0)
        if personal_weight is not None:
            gate = float(personal_weight)
        mixture = (1 - gate) * pooled + gate * c
        coefficients = (1 - gate) * weights.copy()
        coefficients[i] += gate
        effective_sigma = float(np.sqrt(np.sum(coefficients**2 * sigmas**2)))
        ridge = max(config.ridge_floor,
                    config.noise_ridge_multiplier * effective_sigma * np.sqrt(len(c)))
        if fixed_ridge is not None:
            ridge = float(fixed_ridge)
        joint = psd_projection(mixture)
        selected, trace = conditional_select(joint, config.k, ridge, conditional)
        selected_sets.append(selected)
        recovered.append(joint)
        traces.append(trace)
        diagnostics.append(dict(client=i, personal_weight=gate,
                                observed_contrast_energy=observed_energy,
                                estimated_noise_energy=noise_energy, ridge=ridge,
                                effective_sigma=effective_sigma,
                                psd_projection_distance=float(np.linalg.norm(joint - mixture, "fro"))))
    return dict(selected_sets=selected_sets, recovered=recovered, diagnostics=diagnostics,
                traces=traces, centered_moments=moments, pooled=pooled)


def recover_blockwise(h_list: list[np.ndarray], sigmas: np.ndarray, weights: np.ndarray,
                      config: TaskAwareConfig, *, personal_weight: float | None = None,
                      centered: bool = True, conditional: bool = True,
                      fixed_ridge: float | None = None, coordinatewise: bool = True) -> dict:
    """Experimental shared-A, coordinate-shrunk-q recovery from the SAME release.

    Positive-part coordinate shrinkage uses DP noise only, not sampling noise.
    It is an exploratory estimator, not a guaranteed personalization benefit.
    A shared ridge isolates q personalization from regularization changes.
    """
    reference = recover_task_aware(h_list, sigmas, weights, config,
                                   personal_weight=0, centered=centered,
                                   fixed_ridge=fixed_ridge)
    weights = np.asarray(weights, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if personal_weight is not None and not 0 <= personal_weight <= 1:
        raise ValueError("Personal weight must be in [0, 1]")
    moments = reference["centered_moments"]
    variances = [center_release(h, s, centered)[1] for h, s in zip(h_list, sigmas)]
    pool = reference["pooled"]
    pooled_var = sum(w**2 * v for w, v in zip(weights, variances))
    a, qpool = psd_projection(pool[:-1, :-1]), pool[:-1, -1]
    ridge = reference["diagnostics"][0]["ridge"]
    selections, task_vectors, diagnostics, traces = [], [], [], []
    for i, (c, v) in enumerate(zip(moments, variances)):
        delta = c[:-1, -1] - qpool
        variance = np.maximum(v + pooled_var - 2 * weights[i] * v, 0)
        gate = np.clip(1 - variance / np.maximum(delta**2, 1e-30), 0, 1)
        gate[delta == 0] = 0
        if not coordinatewise:
            energy = float(delta @ delta)
            gate[:] = np.clip(1 - variance.sum() / energy, 0, 1) if energy > 0 else 0
        if personal_weight is not None:
            gate[:] = personal_weight
        q = qpool + gate * delta
        selected, trace = conditional_select_blocks(a, q, config.k, ridge, conditional)
        selections.append(selected)
        task_vectors.append(q)
        traces.append(trace)
        diagnostics.append(dict(client=i, personal_weight=float(gate.mean()),
                                active_personal_coordinates=int(np.count_nonzero(gate)),
                                observed_contrast_energy=float(delta @ delta),
                                estimated_noise_energy=float(variance.sum()), ridge=ridge,
                                coordinate_weights=gate.tolist()))
    return dict(selected_sets=selections, feature_covariance=a, task_vectors=task_vectors,
                diagnostics=diagnostics, traces=traces, centered_moments=moments, pooled=pool)


def recover_guarded(h_list: list[np.ndarray], sigmas: np.ndarray, weights: np.ndarray,
                    config: TaskAwareConfig, *, family_alpha: float = .05,
                    max_replacements: int | None = None,
                    personalized: bool = True, centered: bool = True,
                    denoise_shared: bool = False,
                    fixed_ridge: float | None = None) -> dict:
    """Shared-anchored personalization guarded against coordinatewise DP noise.

    Client-minus-pool contrasts are soft-thresholded using a simultaneous
    Gaussian threshold. Pooled q is preserved by default; shared soft
    thresholding is an opt-in ablation. Personalization starts from one common
    subset and may replace only a small public number of its features. This is
    a conservative experimental rule, not a coverage or utility theorem: the
    centered release is nonlinear and sampling variance is not modeled here.
    """
    if not 0 < family_alpha < 1:
        raise ValueError("family_alpha must be in (0, 1)")
    reference = recover_task_aware(h_list, sigmas, weights, config,
                                   personal_weight=0, centered=centered,
                                   fixed_ridge=fixed_ridge)
    weights, sigmas = np.asarray(weights, dtype=float), np.asarray(sigmas, dtype=float)
    moments = reference["centered_moments"]
    variances = [center_release(h, s, centered)[1] for h, s in zip(h_list, sigmas)]
    pool = reference["pooled"]
    pooled_var = sum(w**2 * v for w, v in zip(weights, variances))
    a = psd_projection(pool[:-1, :-1])
    qpool = pool[:-1, -1]
    d = len(qpool)
    z_value = NormalDist().inv_cdf(1 - family_alpha / (2 * d))
    qshared = (soft_threshold(qpool, z_value * np.sqrt(pooled_var))
               if denoise_shared else qpool.copy())
    ridge = reference["diagnostics"][0]["ridge"]
    shared, shared_trace = conditional_select_blocks(a, qshared, config.k, ridge, True)
    if max_replacements is None:
        max_replacements = max(1, int(round(.2 * config.k)))
    if (isinstance(max_replacements, (bool, np.bool_))
            or not isinstance(max_replacements, (int, np.integer))
            or not 0 <= max_replacements <= config.k):
        raise ValueError("Invalid personalized replacement budget")

    selections, task_vectors, diagnostics, traces = [], [], [], []
    for i, (c, v) in enumerate(zip(moments, variances)):
        delta = c[:-1, -1] - qpool
        contrast_var = np.maximum(v + pooled_var - 2 * weights[i] * v, 0)
        personal_delta = soft_threshold(delta, z_value * np.sqrt(contrast_var))
        active = int(np.count_nonzero(personal_delta))
        q = qshared + personal_delta if personalized else qshared.copy()
        selected, swap_trace = anchored_swaps(
            a, q, shared, ridge, max_replacements,
            enabled=personalized and active > 0,
        )
        selections.append(selected)
        task_vectors.append(q)
        traces.append(dict(shared=shared_trace, swaps=swap_trace))
        diagnostics.append(dict(
            client=i,
            personal_weight=float(np.linalg.norm(personal_delta)
                                  / max(np.linalg.norm(delta), 1e-30)),
            active_personal_coordinates=active,
            selected_replacements=len(swap_trace),
            simultaneous_z=float(z_value),
            denoise_shared=bool(denoise_shared),
            max_replacements=int(max_replacements),
            observed_contrast_energy=float(delta @ delta),
            estimated_noise_energy=float(contrast_var.sum()),
            ridge=ridge,
            coordinate_weights=(personal_delta != 0).astype(float).tolist(),
        ))
    return dict(selected_sets=selections, feature_covariance=a, task_vectors=task_vectors,
                diagnostics=diagnostics, traces=traces, centered_moments=moments,
                pooled=pool, shared_selected=shared, shared_task_vector=qshared)
