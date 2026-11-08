"""Conditional-residual candidates and independently calibrated deployment.

Residual gates are plug-in heuristics. The calibration guard, separately,
compares fixed DP-trained classifiers on an independent calibration partition.
It does not certify a subsequently refitted or otherwise different classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from dpspsr.task_aware import (TaskAwareConfig, conditional_select_blocks,
                              anchored_swaps, psd_projection, recover_task_aware)


def conditional_residual(local: np.ndarray, pooled: np.ndarray,
                         beta: np.ndarray) -> np.ndarray:
    """Remove covariance transport from a local-minus-shared task contrast."""
    local, pooled, beta = (np.asarray(x, dtype=float) for x in (local, pooled, beta))
    if (beta.ndim != 1 or local.shape != (len(beta)+1, len(beta)+1)
            or pooled.shape != local.shape
            or not all(np.isfinite(x).all() for x in (local, pooled, beta))):
        raise ValueError("Invalid centered moments or shared coefficients")
    return local[:-1, -1] - pooled[:-1, -1] - (local[:-1, :-1]-pooled[:-1, :-1]) @ beta


def residual_noise_variance(h: np.ndarray, sigma: float, beta: np.ndarray) -> np.ndarray:
    """Fixed-beta variance of centered q-A*beta under symmetric Gaussian noise.

    Evaluating this at noisy means and a beta estimated from the SAME release
    is only a plug-in approximation. It excludes sampling variance and beta
    estimation uncertainty; it is not a confidence interval.
    """
    means, v = h[:-1, -1], sigma**2 / 2
    mx, my = means[:-1], means[-1]
    norm2 = float(beta @ beta)
    e = my - mx @ beta
    linear = 1 + norm2 + beta**2 + e**2 - 2*e*mx*beta + mx**2*(1+norm2)
    return np.maximum(v*linear + v**2*(1+norm2+beta**2), 0)


def covariance_noise_energy(h: np.ndarray, sigma: float) -> float:
    means, v = h[:-2, -1], sigma**2 / 2
    sq = np.maximum(means**2-v, 0)
    var = v*(1+sq[:, None]+sq[None, :]) + v**2
    np.fill_diagonal(var, sigma**2+4*v*sq+2*v**2)
    return float(var.sum())


def recover_residual(h_list: list[np.ndarray], sigmas: np.ndarray, weights: np.ndarray,
                     config: TaskAwareConfig, *, personalize_covariance: bool = True,
                     personalized: bool = True, fixed_ridge: float | None = None) -> dict:
    """Use a group gate on conditional, rather than marginal, task differences.

    No raw observations or validation labels enter candidate construction.
    Both covariance and residual gates are empirical estimators, not utility
    guarantees. All previous algorithms remain available as controls.
    """
    ref = recover_task_aware(h_list, sigmas, weights, config, personal_weight=0,
                             fixed_ridge=fixed_ridge)
    weights, sigmas = np.asarray(weights, float), np.asarray(sigmas, float)
    moments, pool = ref["centered_moments"], ref["pooled"]
    a, q = psd_projection(pool[:-1, :-1]), pool[:-1, -1]
    ridge = ref["diagnostics"][0]["ridge"]
    beta = np.linalg.solve(a+ridge*np.eye(len(q)), q)
    rv = [residual_noise_variance(h, s, beta) for h, s in zip(h_list, sigmas)]
    av = np.array([covariance_noise_energy(h, s) for h, s in zip(h_list, sigmas)])
    pool_rv, pool_av = sum(w*w*v for w, v in zip(weights, rv)), float(weights**2 @ av)
    shared, _ = conditional_select_blocks(a, q, config.k, ridge)
    selections, recovered, diagnostics, traces = [], [], [], []
    for i, c in enumerate(moments):
        r = conditional_residual(c, pool, beta)
        variance = np.maximum(rv[i]+pool_rv-2*weights[i]*rv[i], 0)
        renergy, anoise = float(r @ r), max(av[i]+pool_av-2*weights[i]*av[i], 0)
        aenergy = float(np.sum((c[:-1, :-1]-pool[:-1, :-1])**2))
        gate = float(max(0, 1-float(variance.sum())/renergy)) if renergy > 0 else 0.
        agate = float(max(0, 1-anoise/aenergy)) if aenergy > 0 else 0.
        if not personalized:
            gate = agate = 0.
        if not personalize_covariance:
            agate = 0.
        ai = (1-agate)*a + agate*psd_projection(c[:-1, :-1])
        qi = q + (ai-a) @ beta + gate*r
        if gate == 0 and agate == 0:
            selected, trace = shared.copy(), []
        else:
            selected, trace = conditional_select_blocks(ai, qi, config.k, ridge)
        joint = pool.copy()
        joint[:-1, :-1], joint[:-1, -1], joint[-1, :-1] = ai, qi, qi
        selections.append(selected); recovered.append(joint); traces.append(trace)
        diagnostics.append(dict(client=i, personal_weight=gate, covariance_weight=agate,
            ridge=ridge, conditional_residual_energy=renergy,
            estimated_residual_noise_energy=float(variance.sum()),
            marginal_contrast_energy=float(np.sum((c[:-1, -1]-q)**2)),
            selected_replacements=len(set(selected)-set(shared)),
            exact_shared_fallback=gate == 0 and agate == 0))
    return dict(selected_sets=selections, recovered=recovered, diagnostics=diagnostics,
                traces=traces, pooled=pool, centered_moments=moments, shared_selected=shared)


def recover_shared_core_aux(h_list: list[np.ndarray], sigmas: np.ndarray,
                            weights: np.ndarray, config: TaskAwareConfig, *,
                            max_personal_weight: float = .25,
                            max_replacements: int = 1,
                            fixed_ridge: float | None = None) -> dict:
    """Keep shared structure primary; allow a bounded residual to replace few slots.

    Covariance and ridge remain shared. The local conditional residual is
    reliability-shrunk, capped, and can change only ``max_replacements`` items
    in the shared top-K set. This is a conservative candidate, not a utility
    guarantee; the release-noise gate remains a plug-in estimate.
    """
    if (not np.isfinite(max_personal_weight) or not 0 <= max_personal_weight <= .25
            or isinstance(max_replacements, (bool, np.bool_))
            or not isinstance(max_replacements, (int, np.integer))
            or not 0 <= max_replacements <= config.k):
        raise ValueError("Personal weight cap must be in [0, .25] and replacement cap within K")
    ref = recover_task_aware(h_list, sigmas, weights, config, personal_weight=0,
                             fixed_ridge=fixed_ridge)
    weights, sigmas = np.asarray(weights, float), np.asarray(sigmas, float)
    moments, pool = ref["centered_moments"], ref["pooled"]
    a = psd_projection(pool[:-1, :-1])
    q = pool[:-1, -1]
    ridge = ref["diagnostics"][0]["ridge"]
    beta = np.linalg.solve(a + ridge * np.eye(len(q)), q)
    rv = [residual_noise_variance(h, s, beta) for h, s in zip(h_list, sigmas)]
    pool_rv = sum(w*w*v for w, v in zip(weights, rv))
    shared, shared_trace = conditional_select_blocks(a, q, config.k, ridge)
    selections, recovered, diagnostics, traces = [], [], [], []
    for i, c in enumerate(moments):
        residual = conditional_residual(c, pool, beta)
        variance = np.maximum(rv[i] + pool_rv - 2 * weights[i] * rv[i], 0)
        energy, noise_energy = float(residual @ residual), float(variance.sum())
        reliability = max(0., 1. - noise_energy / energy) if energy > 0 else 0.
        weight = min(max_personal_weight, max_personal_weight * reliability)
        qi = q + weight * residual
        selected, swaps = anchored_swaps(a, qi, shared, ridge, max_replacements,
                                         enabled=weight > 0 and max_replacements > 0)
        joint = pool.copy()
        joint[:-1, :-1], joint[:-1, -1], joint[-1, :-1] = a, qi, qi
        selections.append(selected)
        recovered.append(joint)
        traces.append(dict(shared=shared_trace, swaps=swaps))
        diagnostics.append(dict(client=i, personal_weight=weight,
            reliability_gate=reliability, residual_energy=energy,
            estimated_residual_noise_energy=noise_energy,
            selected_replacements=len(swaps), max_replacements=max_replacements,
            max_personal_weight=max_personal_weight,
            shared_core_preserved=config.k-len(swaps)))
    return dict(selected_sets=selections, recovered=recovered, diagnostics=diagnostics,
                traces=traces, pooled=pool, centered_moments=moments,
                shared_selected=shared)


@dataclass(frozen=True)
class ReleasedRidgeClassifier:
    """Classifier fitted only from a released moment; no private refitting."""
    subset: np.ndarray
    coefficients: np.ndarray
    intercept: float
    clip_norm: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Need finite feature rows")
        scales = np.minimum(1., np.sqrt(self.clip_norm**2-2)
                            / np.maximum(np.linalg.norm(x, axis=1), 1e-12))
        return ((x[:, self.subset]*scales[:, None]) @ self.coefficients + self.intercept >= 0).astype(int)


def fit_released_classifier(h: np.ndarray, subset: np.ndarray, ridge: float,
                             clip_norm: float) -> ReleasedRidgeClassifier:
    """Use local DP moments for every candidate, keeping the learner fixed."""
    h, subset = np.asarray(h, float), np.asarray(subset)
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or len(h) < 3
            or not np.isfinite(h).all() or subset.ndim != 1
            or not np.issubdtype(subset.dtype, np.integer) or not len(subset)
            or len(set(subset.tolist())) != len(subset) or np.any(subset < 0)
            or np.any(subset >= len(h)-2) or not np.isfinite(ridge) or ridge <= 0
            or not np.isfinite(clip_norm) or clip_norm <= np.sqrt(2)):
        raise ValueError("Invalid classifier inputs")
    means = h[:-1, -1]
    c = h[:-1, :-1]-np.outer(means, means)
    a = psd_projection(c[np.ix_(subset, subset)])
    beta = np.linalg.solve(a+ridge*np.eye(len(subset)), c[subset, -1])
    return ReleasedRidgeClassifier(subset.copy(), beta, float(means[-1]-means[subset] @ beta), clip_norm)


def macro_f1_interval(confusion: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Monotone interval extension for fixed two-class macro-F1, zero_division=0.

    Last axis is [TN, FP, FN, TP]; component intervals need not sum to one.
    """
    confusion = np.asarray(confusion, float)
    if (confusion.ndim != 2 or confusion.shape[1] != 4
            or not np.isfinite(confusion).all() or not np.isfinite(radius) or radius < 0):
        raise ValueError("Invalid confusion frequencies or radius")
    lo, hi = np.clip(confusion-radius, 0, 1), np.clip(confusion+radius, 0, 1)
    def value(tn, tp, errors):
        return (np.divide(tn, 2*tn+errors, out=np.zeros_like(tn), where=2*tn+errors > 0)
                + np.divide(tp, 2*tp+errors, out=np.zeros_like(tp), where=2*tp+errors > 0))
    lower = value(lo[:, 0], lo[:, 3], hi[:, 1]+hi[:, 2])
    upper = value(hi[:, 0], hi[:, 3], lo[:, 1]+lo[:, 2])
    return lower, upper


def private_calibration_guard(predictions: np.ndarray, labels: np.ndarray,
                               epsilon: float, delta: float, rng: np.random.Generator,
                               *, alpha: float = .05, min_gain: float = 0.) -> dict:
    """Select only when candidate lower macro-F1 exceeds shared upper macro-F1.

    Candidate 0 is shared. Predictions are from fixed models independent of
    these iid calibration rows. One joint Gaussian release of confusion
    frequencies has replace-one sensitivity sqrt(2*M)/n. Sampling Hoeffding
    and Gaussian union bounds each receive alpha/2. With probability >=1-alpha
    an accepted classifier improves population macro-F1 by > min_gain UNDER
    those assumptions, not necessarily the realized evaluation-set macro-F1.

    Public fixed calibration size, fixed partition membership and no model
    refitting are required. alpha is per client/run, not across all experiments.
    Seeded simulation RNG and raw calibration diagnostics are not deployment DP.
    """
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if (predictions.ndim != 2 or labels.ndim != 1 or predictions.shape[0] != len(labels)
            or not len(labels) or predictions.shape[1] < 1
            or not np.isin(predictions, [0, 1]).all() or not np.isin(labels, [0, 1]).all()
            or not 0 < epsilon <= 1 or not 0 < delta < 1 or not 0 < alpha < 1
            or not np.isfinite(min_gain) or min_gain < 0):
        raise ValueError("Invalid calibration inputs")
    n, m = predictions.shape
    confusion = np.stack([(2*labels[:, None]+predictions == cell).mean(axis=0)
                           for cell in range(4)], axis=1)
    sensitivity = np.sqrt(2*m)/n
    sigma = sensitivity*np.sqrt(2*np.log(1.25/delta))/epsilon
    released = confusion+rng.normal(0, sigma, size=confusion.shape)
    z = NormalDist().inv_cdf(1-alpha/(16*m))
    sampling_radius = np.sqrt(np.log(16*m/alpha)/(2*n))
    radius = sampling_radius+z*sigma
    lower, upper = macro_f1_interval(released, radius)
    best = int(np.argmax(lower))
    accepted = best != 0 and lower[best] > upper[0]+min_gain
    return dict(selected_index=best if accepted else 0, accepted=bool(accepted),
                lower=lower.tolist(), upper=upper.tolist(),
                released_confusion=released.tolist(), sampling_radius=float(sampling_radius),
                noise_sigma=float(sigma), total_radius=float(radius),
                sensitivity=float(sensitivity), alpha=alpha, candidate_count=m)
