from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import mutual_info_score


@dataclass(frozen=True)
class SelectionResult:
    indices: np.ndarray
    scores: np.ndarray
    metadata: dict[str, object]


def validate_selection(indices: np.ndarray, selected_k: int, dimension: int) -> np.ndarray:
    selected = np.asarray(indices, dtype=int).reshape(-1)
    if len(selected) != selected_k:
        raise ValueError(f"Expected {selected_k} features, received {len(selected)}")
    if len(np.unique(selected)) != selected_k:
        raise ValueError("Selected feature indices are not unique")
    if np.any(selected < 0) or np.any(selected >= dimension):
        raise ValueError("Selected feature index is outside the encoded feature space")
    return selected


def _discretize_column(values: np.ndarray, bins: int, categorical: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    unique_values, inverse = np.unique(values, return_inverse=True)
    if categorical or len(unique_values) <= bins:
        return inverse.astype(np.int32)

    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int32)
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)


def select_mrmr_mid(
    features: np.ndarray,
    labels: np.ndarray,
    selected_k: int,
    feature_names: list[str],
    bins: int = 10,
) -> SelectionResult:
    """Centralized mRMR-MID using training-only discrete mutual information.

    Numerical columns are quantile-discretized and one-hot categorical columns
    retain their observed discrete levels. The greedy score is relevance minus
    mean redundancy, matching the standard MID form.
    """
    x = np.asarray(features, dtype=float)
    y = (np.asarray(labels).reshape(-1) > 0).astype(np.int32)
    n, d = x.shape
    if len(y) != n or len(feature_names) != d:
        raise ValueError("mRMR input dimensions do not match")
    if not 0 < selected_k <= d:
        raise ValueError("selected_k must be between 1 and the feature dimension")

    discrete = np.empty((n, d), dtype=np.int32)
    for feature in range(d):
        is_categorical = feature_names[feature].startswith("cat__")
        discrete[:, feature] = _discretize_column(
            x[:, feature], bins=bins, categorical=is_categorical
        )

    relevance = np.asarray(
        [mutual_info_score(discrete[:, feature], y) for feature in range(d)],
        dtype=float,
    )
    selected: list[int] = [int(np.argmax(relevance))]
    step_scores = np.full((selected_k, d), np.nan, dtype=float)
    step_scores[0] = relevance
    redundancy_sum = np.zeros(d, dtype=float)

    while len(selected) < selected_k:
        newest = selected[-1]
        for candidate in range(d):
            if candidate not in selected:
                redundancy_sum[candidate] += mutual_info_score(
                    discrete[:, candidate], discrete[:, newest]
                )
        score = relevance - redundancy_sum / len(selected)
        score[np.asarray(selected, dtype=int)] = -np.inf
        step_scores[len(selected)] = score
        selected.append(int(np.argmax(score)))

    indices = validate_selection(np.asarray(selected), selected_k, d)
    return SelectionResult(
        indices=indices,
        scores=step_scores,
        metadata={
            "method": "Centralized mRMR-MID",
            "input_scope": "pooled training records",
            "privacy": "non-private",
            "numeric_discretization": f"{bins}-quantile bins",
            "categorical_discretization": "observed one-hot levels",
            "criterion": "mutual information relevance minus mean redundancy",
        },
    )


def select_centralized_chi2(
    features: np.ndarray,
    labels: np.ndarray,
    selected_k: int,
) -> SelectionResult:
    """Centralized non-private chi-square ranking on pooled training records.

    Chi-square requires nonnegative inputs. The pooled training matrix is
    transformed feature-wise to [0, 1] using extrema computed from that same
    training matrix. The transformation and ranking never access test data.
    """
    x = np.asarray(features, dtype=float)
    y = (np.asarray(labels).reshape(-1) > 0).astype(np.int32)
    if x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError("Centralized Chi2 input dimensions do not match")
    n, d = x.shape
    if not 0 < selected_k <= d:
        raise ValueError("selected_k must be between 1 and the feature dimension")
    if not np.isfinite(x).all():
        raise ValueError("Centralized Chi2 received non-finite features")

    feature_min = np.min(x, axis=0)
    feature_max = np.max(x, axis=0)
    span = feature_max - feature_min
    safe_span = np.where(span > 0.0, span, 1.0)
    scaled = np.clip((x - feature_min) / safe_span, 0.0, 1.0)
    scores, _ = chi2(scaled, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    ranking = np.lexsort((np.arange(d), -scores))
    indices = validate_selection(ranking[:selected_k], selected_k, d)
    return SelectionResult(
        indices=indices,
        scores=scores,
        metadata={
            "method": "Centralized Chi2",
            "input_scope": "pooled training records",
            "privacy": "non-private",
            "normalization": "pooled-training feature-wise min-max to [0,1]",
            "criterion": "sklearn.feature_selection.chi2",
            "selection_rule": "deterministic exact top-K with feature-index tie break",
            "training_rows": n,
        },
    )


def select_centralized_mi(
    features: np.ndarray,
    labels: np.ndarray,
    selected_k: int,
    feature_names: list[str],
    seed: int,
    n_neighbors: int = 3,
) -> SelectionResult:
    """Centralized non-private mutual-information ranking.

    Encoded categorical columns are treated as discrete and standardized
    numerical columns as continuous. The estimator is seeded and the exact
    top-K ranking uses feature index as a deterministic tie break.
    """
    x = np.asarray(features, dtype=float)
    y = (np.asarray(labels).reshape(-1) > 0).astype(np.int32)
    if x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError("Centralized MI input dimensions do not match")
    n, d = x.shape
    if len(feature_names) != d:
        raise ValueError("Centralized MI feature names do not match the feature dimension")
    if not 0 < selected_k <= d:
        raise ValueError("selected_k must be between 1 and the feature dimension")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    if not np.isfinite(x).all():
        raise ValueError("Centralized MI received non-finite features")

    discrete_mask = np.asarray(
        [name.startswith("cat__") for name in feature_names], dtype=bool
    )
    scores = mutual_info_classif(
        x,
        y,
        discrete_features=discrete_mask,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    ranking = np.lexsort((np.arange(d), -scores))
    indices = validate_selection(ranking[:selected_k], selected_k, d)
    return SelectionResult(
        indices=indices,
        scores=scores,
        metadata={
            "method": "Centralized MI",
            "input_scope": "pooled training records",
            "privacy": "non-private",
            "criterion": "sklearn.feature_selection.mutual_info_classif",
            "discrete_features": "encoded cat__ columns",
            "continuous_features": "encoded num__ columns",
            "n_neighbors": n_neighbors,
            "seed": seed,
            "selection_rule": "exact top-K with feature-index tie break",
            "training_rows": n,
        },
    )


def select_federated_chi2(
    client_features: list[np.ndarray],
    client_labels: list[np.ndarray],
    selected_k: int,
) -> SelectionResult:
    """Training-only federated chi-square ranking inspired by Fed-FeRe.

    Clients share feature-wise extrema and sample-normalized chi-square scores,
    not raw records. The server forms a sample-weighted aggregate and returns
    one exact-K global subset. This is a controlled adaptation of Fed-FeRe's
    chi-square reduction principle, not a reproduction of its GRU/FedAvg
    training framework or adaptive threshold rule.
    """
    if not client_features or len(client_features) != len(client_labels):
        raise ValueError("Federated chi-square requires matched client arrays")

    features = [np.asarray(values, dtype=float) for values in client_features]
    labels = [
        (np.asarray(values).reshape(-1) > 0).astype(np.int32)
        for values in client_labels
    ]
    dimension = features[0].shape[1]
    if not 0 < selected_k <= dimension:
        raise ValueError("selected_k must be between 1 and the feature dimension")
    for x, y in zip(features, labels):
        if x.ndim != 2 or x.shape[1] != dimension or x.shape[0] != len(y):
            raise ValueError("Federated chi-square client dimensions do not match")
        if not np.isfinite(x).all():
            raise ValueError("Federated chi-square received non-finite features")

    global_min = np.min(np.vstack([np.min(x, axis=0) for x in features]), axis=0)
    global_max = np.max(np.vstack([np.max(x, axis=0) for x in features]), axis=0)
    span = global_max - global_min
    safe_span = np.where(span > 0.0, span, 1.0)

    sample_counts = np.asarray([len(y) for y in labels], dtype=float)
    weights = sample_counts / sample_counts.sum()
    local_scores = np.zeros((len(features), dimension), dtype=float)
    for client, (x, y) in enumerate(zip(features, labels)):
        scaled = np.clip((x - global_min) / safe_span, 0.0, 1.0)
        scores, _ = chi2(scaled, y)
        local_scores[client] = np.nan_to_num(
            scores / max(len(y), 1), nan=0.0, posinf=0.0, neginf=0.0
        )

    aggregate_scores = weights @ local_scores
    ranking = np.lexsort((np.arange(dimension), -aggregate_scores))
    indices = validate_selection(ranking[:selected_k], selected_k, dimension)
    return SelectionResult(
        indices=indices,
        scores=aggregate_scores,
        metadata={
            "method": "Federated Chi2 (Fed-FeRe-inspired)",
            "input_scope": "client-local training records; aggregated scores only",
            "privacy": "non-private; no formal DP guarantee",
            "normalization": "training-only global min-max from client extrema",
            "local_score": "sklearn chi2 divided by local sample count",
            "aggregation": "sample-count-weighted mean of normalized local scores",
            "selection_rule": "exact top-K; replaces Fed-FeRe adaptive threshold for budget matching",
            "reproduction_scope": "controlled feature-reduction adaptation, not end-to-end Fed-FeRe",
        },
    )


def _scaled_abs_kendall(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    n, d = features.shape
    coefficients = np.empty(d, dtype=float)
    for feature in range(d):
        tau = stats.kendalltau(
            features[:, feature],
            target,
            method="asymptotic",
            variant="b",
            alternative="two-sided",
        ).statistic
        coefficients[feature] = 0.0 if not np.isfinite(tau) else abs((n / 2.0) * tau)
    return coefficients


def select_dp_kendall_official(
    features: np.ndarray,
    labels: np.ndarray,
    selected_k: int,
    epsilon: float,
    seed: int,
    tie_jitter: float = 1e-6,
) -> SelectionResult:
    """DPKendall adapted from the author-released Google Research algorithm.

    The mechanism uses add/remove adjacency, sensitivity 3/2 in the first
    round and 3 thereafter, epsilon/K sequential composition, and the basic
    Gumbel peeling mechanism. Tiny seeded jitter follows the authors' data
    preparation guidance for tied values.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    n, d = x.shape
    if len(y) != n:
        raise ValueError("DPKendall input dimensions do not match")
    if not 0 < selected_k <= d:
        raise ValueError("selected_k must be between 1 and the feature dimension")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    rng = np.random.RandomState(seed)
    if tie_jitter > 0:
        x = x + rng.uniform(-tie_jitter, tie_jitter, size=x.shape)
        y = y + rng.uniform(-tie_jitter, tie_jitter, size=y.shape)

    label_coefficients = _scaled_abs_kendall(x, y)
    redundancy_sum = np.zeros(d, dtype=float)
    selected: list[int] = []
    step_scores = np.full((selected_k, d), np.nan, dtype=float)
    split_epsilon = epsilon / selected_k

    for step in range(selected_k):
        sensitivity = 1.5 if step == 0 else 3.0
        score = label_coefficients.copy()
        if selected:
            redundancy_sum += _scaled_abs_kendall(x, x[:, selected[-1]])
            score -= redundancy_sum / len(selected)
        if selected:
            score[np.asarray(selected, dtype=int)] = -np.inf

        # The official basic peeling mechanism halves epsilon for a
        # non-monotone score before calibrating Gumbel noise.
        local_epsilon = split_epsilon / 2.0
        noise_scale = sensitivity / local_epsilon
        noisy_score = score + rng.gumbel(scale=noise_scale, size=d)
        chosen = int(np.argmax(noisy_score))
        selected.append(chosen)
        step_scores[step] = score

    indices = validate_selection(np.asarray(selected), selected_k, d)
    return SelectionResult(
        indices=indices,
        scores=step_scores,
        metadata={
            "method": "DPKendall",
            "input_scope": "pooled training records",
            "privacy": f"epsilon-DP with epsilon={epsilon}",
            "adjacency": "add/remove one record",
            "round_budget": epsilon / selected_k,
            "first_round_sensitivity": 1.5,
            "later_round_sensitivity": 3.0,
            "tie_jitter": tie_jitter,
            "seed": seed,
            "source": "google-research/private_kendall",
        },
    )


def mean_abs_pairwise_redundancy(features: np.ndarray, indices: np.ndarray) -> float:
    selected = np.asarray(indices, dtype=int)
    if len(selected) < 2:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(
            np.asarray(features, dtype=float)[:, selected], rowvar=False
        )
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.triu_indices(len(selected), k=1)
    return float(np.mean(np.abs(corr[upper])))
