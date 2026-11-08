"""Tree-structured shared-to-personalized feature selection.

This adapts the global-tree / monotone-refinement idea of FedTreeLoRA to
second-order feature-selection releases. All client comparisons are
post-processing of the existing DP releases. Noise gates are plug-in
heuristics, not utility or inference guarantees.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from dpspsr.personalization import residual_noise_variance
from dpspsr.task_aware import (TaskAwareConfig, conditional_select_blocks,
                               psd_projection, recover_task_aware)


def _conditional_profile(a: np.ndarray, q: np.ndarray, selected: list[int],
                         ridge: float) -> np.ndarray:
    if not selected:
        return q.copy()
    ids = np.asarray(selected, dtype=int)
    ass = a[np.ix_(ids, ids)] + ridge * np.eye(len(ids))
    return q - a[:, ids] @ np.linalg.solve(ass, q[ids])


def _next_feature(a: np.ndarray, q: np.ndarray, selected: list[int],
                  ridge: float, conditional: bool = True) -> int:
    residual_q = q.copy()
    residual_var = np.diag(a).copy() + ridge
    if selected and conditional:
        ids = np.asarray(selected, dtype=int)
        ass = a[np.ix_(ids, ids)] + ridge*np.eye(len(ids))
        cross = a[:, ids]
        residual_q -= cross @ np.linalg.solve(ass, q[ids])
        residual_var -= np.sum(cross*np.linalg.solve(ass, cross.T).T, axis=1)
    scores = residual_q**2/np.maximum(residual_var, ridge)
    scores[np.asarray(selected, dtype=int)] = -np.inf
    return int(np.argmax(scores))


def _distance(profiles: list[np.ndarray], variances: list[np.ndarray]) -> np.ndarray:
    n = len(profiles)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i):
            scale = np.maximum(variances[i] + variances[j], 1e-12)
            out[i, j] = out[j, i] = np.sqrt(np.mean((profiles[i]-profiles[j])**2/scale))
    return out


def _cut(tree: np.ndarray, n_clients: int, clusters: int) -> np.ndarray:
    if clusters == 1:
        return np.zeros(n_clients, dtype=int)
    return cut_tree(tree, n_clusters=[clusters]).reshape(-1).astype(int)


def recover_tree_hierarchical(h_list: list[np.ndarray], sigmas: np.ndarray,
                              weights: np.ndarray, config: TaskAwareConfig, *,
                              shared_fraction: float = .7, max_clusters: int | None = None,
                              split_window: int = 1, silhouette_threshold: float = 0.,
                              max_edge_weight: float = .25,
                              fixed_ridge: float | None = None,
                              has_public_intercept: bool = True,
                              conditional: bool = True) -> dict:
    """Select a shared feature trunk, then refine it along one global client tree.

    A single average-linkage tree is built from noise-standardized conditional
    client profiles. Its cuts refine monotonically as feature selection
    proceeds. A fixed initial prefix is selected at the root; later branches
    inherit all parent features and add their own choices. Branch task vectors
    receive a reliability-shrunk edge residual, capped by ``max_edge_weight``.
    The feature covariance and ridge remain global at every node.
    """
    weights, sigmas = np.asarray(weights, float), np.asarray(sigmas, float)
    n = len(h_list)
    if (n < 2 or weights.shape != (n,) or sigmas.shape != (n,)
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1) or not np.isfinite(sigmas).all()
            or np.any(sigmas < 0) or not 0 < shared_fraction <= 1
            or not isinstance(split_window, (int, np.integer)) or split_window < 1
            or not 0 <= silhouette_threshold < 1
            or not 0 <= max_edge_weight <= 1
            or not isinstance(has_public_intercept, (bool, np.bool_))
            or not isinstance(conditional, (bool, np.bool_))
            or (max_clusters is not None and (not isinstance(max_clusters, (int, np.integer))
                                               or not 1 <= max_clusters <= n))):
        raise ValueError("Invalid tree recovery configuration")
    if has_public_intercept:
        ref = recover_task_aware(h_list, sigmas, weights, config, personal_weight=0,
                                 fixed_ridge=fixed_ridge)
        moments, pool = ref["centered_moments"], ref["pooled"]
        ridge = ref["diagnostics"][0]["ridge"]
    else:
        # Compatibility path for the frozen [x,y] release: no sample means
        # are available, so recover from raw second moments without centering.
        moments = [np.asarray(h, dtype=float) for h in h_list]
        pool = sum(w * h for w, h in zip(weights, moments))
        pooled_sigma = float(np.sqrt(np.sum(weights**2 * sigmas**2)))
        ridge = max(config.ridge_floor,
                    config.noise_ridge_multiplier * pooled_sigma * np.sqrt(len(pool)))
        if fixed_ridge is not None:
            ridge = float(fixed_ridge)
    a = psd_projection(pool[:-1, :-1])
    q = pool[:-1, -1]
    beta = np.linalg.solve(a + ridge*np.eye(len(q)), q)
    local_profiles = [c[:-1, -1] - c[:-1, :-1] @ beta for c in moments]
    if has_public_intercept:
        local_variances = [residual_noise_variance(h, s, beta)
                           for h, s in zip(h_list, sigmas)]
    else:
        # Plug-in marginal variance of q_i - A_i beta for symmetric Gaussian
        # noise on the legacy [x,y] second-moment release.
        local_variances = [np.maximum(
            s**2 * (.5 * (1 + float(beta @ beta)) + .5 * beta**2), 1e-12)
            for s in sigmas]

    # One topology is reused at every selection step, so client groups cannot
    # contradict one another as the feature set grows.
    d_global = _distance(local_profiles, local_variances)
    tree = linkage(squareform(d_global, checks=False), method="average") if n > 2 else np.empty((0, 4))
    cap = min(n-1, n if max_clusters is None else max_clusters)
    shared_prefix = min(config.k, max(1, int(np.ceil(shared_fraction*config.k))))
    shared, shared_trace = conditional_select_blocks(
        a, q, shared_prefix, ridge, conditional=conditional
    )
    selected_by_client = [[] for _ in range(n)]
    previous_labels = np.zeros(n, dtype=int)
    previous_count = 1
    previous_raw_profile = {0: sum(w*u for w, u in zip(weights, local_profiles))}
    previous_model_profile = {0: q.copy()}
    selected_tree_counts, selected_silhouettes, edge_weights = [], [], [[] for _ in range(n)]
    traces = [dict(shared=shared_trace, branches=[] ) for _ in range(n)]

    for step in range(config.k):
        if step < shared_prefix:
            labels, count, silhouette = np.zeros(n, dtype=int), 1, silhouette_threshold
            profiles = local_profiles
        else:
            profiles = [_conditional_profile(a, c[:-1, -1], selected_by_client[i], ridge)
                        for i, c in enumerate(moments)]
            d_step = _distance(profiles, local_variances)
            next_max = min(cap, previous_count + split_window)
            candidates = list(range(previous_count, next_max+1))
            candidate_scores = []
            for count_candidate in candidates:
                candidate_labels = _cut(tree, n, count_candidate)
                if count_candidate == 1 or count_candidate >= n or len(np.unique(candidate_labels)) < 2:
                    score = silhouette_threshold
                else:
                    score = float(silhouette_score(d_step, candidate_labels, metric="precomputed"))
                candidate_scores.append((score, -count_candidate, candidate_labels, count_candidate))
            silhouette, _, labels, count = max(candidate_scores, key=lambda z: (z[0], z[1]))
            if count > previous_count and silhouette <= silhouette_threshold:
                labels, count, silhouette = previous_labels.copy(), previous_count, silhouette_threshold

        # Tree cuts are nested. Every child therefore inherits its parent's
        # selected prefix and adds one group-level feature.
        groups = sorted(np.unique(labels).tolist())
        current_raw, current_model, current_selected = {}, {}, {}
        for label in groups:
            members = np.flatnonzero(labels == label)
            parent_labels = np.unique(previous_labels[members])
            if len(parent_labels) != 1:
                raise RuntimeError("Selected tree cuts must refine the previous cut")
            parent = int(parent_labels[0])
            parent_features = selected_by_client[int(members[0])].copy()
            if any(selected_by_client[int(j)] != parent_features for j in members):
                raise RuntimeError("Tree peers must share the inherited feature prefix")
            mass = float(weights[members].sum())
            coeff = weights[members] / mass
            raw_profile = sum(w*local_profiles[int(j)] for w, j in zip(coeff, members))
            if parent == label and count == previous_count:
                model_profile = previous_model_profile[parent].copy()
                edge_weight = 0.
            else:
                parent_members = np.flatnonzero(previous_labels == parent)
                parent_mass = float(weights[parent_members].sum())
                parent_coeff = weights[parent_members] / parent_mass
                parent_raw = previous_raw_profile[parent]
                delta = raw_profile - parent_raw
                coeff_diff = np.zeros(n)
                coeff_diff[members] += coeff
                coeff_diff[parent_members] -= parent_coeff
                delta_variance = sum(coeff_diff[j]**2 * local_variances[j] for j in range(n))
                energy = float(delta @ delta)
                reliability = max(0., 1. - float(np.sum(delta_variance))/energy) if energy > 0 else 0.
                edge_weight = max_edge_weight * reliability
                model_profile = previous_model_profile[parent] + edge_weight*delta
            if len(parent_features) < config.k:
                # Preserve the shared covariance/ridge and use the branch's
                # shrunk task vector only for the next conditional choice.
                parent_features.append(_next_feature(
                    a, model_profile, parent_features, ridge, conditional=conditional
                ))
            current_raw[int(label)] = raw_profile
            current_model[int(label)] = model_profile
            current_selected[int(label)] = parent_features
            for j in members:
                edge_weights[int(j)].append(float(edge_weight))
                traces[int(j)]["branches"].append(dict(step=step+1, cluster=int(label),
                    members=members.tolist(), edge_weight=float(edge_weight),
                    silhouette=float(silhouette)))
                selected_by_client[int(j)] = parent_features.copy()
        selected_tree_counts.append(int(count))
        selected_silhouettes.append(float(silhouette))
        previous_labels, previous_count = labels.copy(), int(count)
        previous_raw_profile, previous_model_profile = current_raw, current_model

    recovered = []
    diagnostics = []
    for i, selected in enumerate(selected_by_client):
        node = int(previous_labels[i])
        joint = pool.copy()
        joint[:-1, :-1], joint[:-1, -1], joint[-1, :-1] = a, previous_model_profile[node], previous_model_profile[node]
        recovered.append(joint)
        diagnostics.append(dict(client=i, personal_weight=max(edge_weights[i], default=0.),
            tree_depth=max(0, previous_count-1), final_cluster_count=previous_count,
            selected_replacements=len(set(selected)-set(shared.tolist())),
            shared_prefix_length=shared_prefix, max_edge_weight=max_edge_weight,
            conditional_redundancy=bool(conditional),
            tree_cluster_counts=selected_tree_counts.copy(),
            tree_silhouettes=selected_silhouettes.copy()))
    return dict(selected_sets=[np.asarray(s, dtype=int) for s in selected_by_client],
                recovered=recovered, diagnostics=diagnostics, traces=traces,
                pooled=pool, centered_moments=moments, shared_selected=shared,
                linkage=tree, cluster_counts=selected_tree_counts,
                silhouettes=selected_silhouettes, global_distance=d_global)
