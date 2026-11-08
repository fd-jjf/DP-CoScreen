"""Paired, training-only mechanism audit. No final holdout evaluation or tuning."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.stats import t
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dpspsr.core import (conditional_score, dp_shared_operator, greedy_select,
                        leading_subspace, population_operator,
                        residual_and_personal_structure,
                        shared_structure_from_subspace)
from dpspsr.fixed_privacy import (FixedFeatureMap, StructurePrivacyProtocol,
                                 empirical_structure, symmetric_noise)
from preprocess_real_e6 import DATASETS, _partition_noniid, _safe_train_test_split


PARAMS = dict(shared_rank=3, personal_rank=2, lambda_s=0.2, lambda_p=0.1,
              alpha=1.0, beta=1.0, gamma=0.4, k=10)
VARIANTS = {
    "Full": "shared rank3 + orthogonal personal rank2 + local redundancy",
    "No-correction": "same as Full with scalar correction omitted",
    "Shared-only": "Full shared structure, zero personal relevance",
    "Personal-only": "no shared subspace; local rank2 soft shrinkage",
    "Local-denoised-rank5": "no sharing; local rank5 soft shrinkage",
    "No-redundancy": "Full with gamma=0",
    "No-disagreement": "Full with lambda_s=0",
    "DP-Local": "local noisy relevance + local noisy redundancy",
    "DP-Global": "pooled noisy relevance + pooled noisy redundancy; common subset",
    "Empirical-reference": "nonprivate Full on fitting empirical structures; NOT population truth",
    "All-features": "all fixed-map coordinates; nonprivate classifier reference",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def recover(h: list[np.ndarray], w: np.ndarray, sigmas: np.ndarray,
            corrected: bool = True, lambda_s: float = 0.2) -> dict:
    operator = (dp_shared_operator(h, w, sigmas, lambda_s) if corrected
                else population_operator(h, w, lambda_s))
    p = leading_subspace(operator, PARAMS["shared_rank"])
    hs = shared_structure_from_subspace(p, h, w)
    personal = [residual_and_personal_structure(hi, p, PARAMS["personal_rank"],
                                               PARAMS["lambda_p"]) for hi in h]
    selected = [greedy_select(hs, item[1], hi, PARAMS["k"], 1, 1, PARAMS["gamma"])[0]
                for hi, item in zip(h, personal)]
    return dict(operator=operator, p=p, hs=hs, personal=personal, selected=selected)


def variant_subsets(h: list[np.ndarray], w: np.ndarray, sigma: np.ndarray) -> tuple[dict, dict, dict]:
    full = recover(h, w, sigma)
    raw = recover(h, w, sigma, corrected=False)
    no_disagreement = recover(h, w, sigma, lambda_s=0)
    zero = np.zeros_like(h[0])
    empty_p = np.empty((h[0].shape[0], 0))
    pooled = sum(wi * hi for wi, hi in zip(w, h))
    global_subset = greedy_select(pooled, zero, pooled, PARAMS["k"], 1, 0, PARAMS["gamma"])[0]
    result = {name: [] for name in VARIANTS if name not in {"Empirical-reference", "All-features"}}
    for i, hi in enumerate(h):
        select = lambda hs, hp, gh: greedy_select(hs, hp, hi, PARAMS["k"], 1, 1, gh)[0]
        personal_only = residual_and_personal_structure(hi, empty_p, 2, PARAMS["lambda_p"])[1]
        rank5 = residual_and_personal_structure(hi, empty_p, 5, PARAMS["lambda_p"])[1]
        for name, subset in {
            "Full": full["selected"][i], "No-correction": raw["selected"][i],
            "Shared-only": select(full["hs"], zero, PARAMS["gamma"]),
            "Personal-only": select(zero, personal_only, PARAMS["gamma"]),
            "Local-denoised-rank5": select(zero, rank5, PARAMS["gamma"]),
            "No-redundancy": select(full["hs"], full["personal"][i][1], 0),
            "No-disagreement": no_disagreement["selected"][i],
            "DP-Local": select(hi, zero, PARAMS["gamma"]),
            "DP-Global": global_subset,
        }.items():
            result[name].append(subset)
    return result, full, raw


def score_diagnostics(full: dict, reference: dict, noisy: list, clean: list,
                      dataset: str, split: int, noise: int) -> list[dict]:
    rows = []
    for i, (hi, h0) in enumerate(zip(noisy, clean)):
        hs, hp = full["hs"], full["personal"][i][1]
        hs0, hp0 = reference["hs"], reference["personal"][i][1]
        prefix: list[int] = []
        for step in range(PARAMS["k"]):
            a = conditional_score(hs, hp, hi, prefix, 1, 1, PARAMS["gamma"])
            b = conditional_score(hs0, hp0, h0, prefix, 1, 1, PARAMS["gamma"])
            bound = (np.max(np.abs(hs[:-1, -1] - hs0[:-1, -1]))
                     + np.max(np.abs(hp[:-1, -1] - hp0[:-1, -1]))
                     + (PARAMS["gamma"] * np.max(np.abs(hi[:-1, :-1] - h0[:-1, :-1])) if prefix else 0))
            finite = np.isfinite(b)
            actual = np.max(np.abs(a[finite] - b[finite]))
            ordered_scores = np.sort(b[finite])
            gap = ordered_scores[-1] - ordered_scores[-2]
            choice, choice0 = int(np.argmax(a)), int(np.argmax(b))
            certified = bool(gap > 2 * bound)
            if actual > bound + 1e-10 or (certified and choice != choice0):
                raise AssertionError("Score-error or margin implication violated")
            rows.append(dict(dataset=dataset, split_seed=split, noise_seed=noise, client=i,
                             step=step, actual_error=float(actual), bound=float(bound),
                             empirical_gap=float(gap), certified=certified,
                             common_prefix_choice_equal=choice == choice0))
            prefix.append(choice0)
    return rows


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    pred = (probability >= 0.5).astype(int)
    return dict(macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                attack_recall=float(recall_score(y, pred, zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                auc=float(roc_auc_score(y, probability)))


def evaluate(xfit: np.ndarray, yfit: np.ndarray, xval: np.ndarray, yval: np.ndarray,
             xpool: np.ndarray, ypool: np.ndarray, subset: np.ndarray) -> dict:
    scaler = StandardScaler().fit(xfit[:, subset])
    xt = scaler.transform(xfit[:, subset])
    model = LogisticRegression(solver="liblinear", max_iter=2000,
                               class_weight="balanced", random_state=2026)
    model.fit(xt, yfit)
    output = {}
    for view, x, y in [("local", xval, yval), ("pooled", xpool, ypool)]:
        probability = expit(model.decision_function(scaler.transform(x[:, subset])))
        output[view] = metrics(y, probability)
    return output


def aggregate(out: Path, metric_rows: list, selection_rows: list, correction_rows: list,
              score_rows: list, mechanism_rows: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {
        "per_client_metrics": pd.DataFrame(metric_rows), "selections": pd.DataFrame(selection_rows),
        "correction_diagnostics": pd.DataFrame(correction_rows),
        "score_bound_diagnostics": pd.DataFrame(score_rows),
        "mechanism_diagnostics": pd.DataFrame(mechanism_rows),
    }
    for filename, frame in frames.items():
        frame.to_csv(out / (filename + ".csv"), index=False)
    metric_names = ["macro_f1", "attack_recall", "balanced_accuracy", "auc"]
    # Clients and paired noise seeds are nested measurements, not independent trials.
    repeats = frames["per_client_metrics"].groupby(
        ["dataset", "split_seed", "variant", "view"], as_index=False)[metric_names].mean()
    repeats.to_csv(out / "repeat_summary.csv", index=False)
    summary_rows, paired_rows = [], []
    for (dataset, variant, view), group in repeats.groupby(["dataset", "variant", "view"]):
        for metric in metric_names:
            values = group[metric].to_numpy()
            summary_rows.append(dict(dataset=dataset, variant=variant, view=view, metric=metric,
                                     mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else 0,
                                     partition_repeats=len(values)))
    for (dataset, view), group in repeats.groupby(["dataset", "view"]):
        full = group[group.variant == "Full"].set_index("split_seed")
        for variant, other in group.groupby("variant"):
            if variant == "Full":
                continue
            other = other.set_index("split_seed").loc[full.index]
            for metric in metric_names:
                delta = full[metric].to_numpy() - other[metric].to_numpy()
                mean = float(delta.mean())
                se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0
                radius = float(t.ppf(0.975, len(delta) - 1) * se) if len(delta) > 1 else None
                paired_rows.append(dict(dataset=dataset, view=view, comparator=variant, metric=metric,
                                        full_minus_comparator=mean,
                                        ci95_low=mean - radius if radius is not None else None,
                                        ci95_high=mean + radius if radius is not None else None,
                                        positive_splits=int(np.count_nonzero(delta > 1e-12)),
                                        partition_repeats=len(delta)))
    summary, paired = pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired_differences.csv", index=False)
    return summary, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--split-start", type=int, default=2101)
    parser.add_argument("--noise-start", type=int, default=9101)
    args = parser.parse_args()
    if args.splits < 1 or args.noise_repeats < 1:
        parser.error("Positive repeat counts are required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = StructurePrivacyProtocol()
    split_seeds = list(range(args.split_start, args.split_start + args.splits))
    noise_seeds = list(range(args.noise_start, args.noise_start + args.noise_repeats))
    root = Path(__file__).resolve().parent
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    privacy=protocol.manifest(), parameters=PARAMS, variants=VARIANTS,
                    split_seeds=split_seeds, noise_seeds=noise_seeds, clients=4,
                    dirichlet_alpha=0.4, reserved_test_split_seed=42, reserved_test_fraction=0.2,
                    client_validation_fraction=0.2, final_holdout_transformed_or_evaluated=False,
                    feature_budget_unit="fixed-map coordinates, NOT raw sensor fields",
                    statistical_unit="partition seed; average clients and paired noise first",
                    interval_scope="exploratory t intervals across overlapping-data repartitions; not population inference",
                    classifier="local StandardScaler+balanced liblinear logistic regression, threshold0.5; not DP output",
                    seed_derivation="SeedSequence([partition_seed,noise_seed,client_id])",
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__, datasets={}, sources={})
    for filename in ["dpspsr/fixed_privacy.py", "dpspsr/core.py", "preprocess_real_e6.py",
                     "test_fixed_privacy.py", "run_fixed_protocol_pilot.py"]:
        manifest["sources"][filename] = sha256(root / filename)
    for name in args.datasets:
        config = DATASETS[name]
        mapper = FixedFeatureMap(config["numeric"], config["categorical"], protocol.categorical_buckets)
        manifest["datasets"][name] = dict(source=str(config["path"]), sha256=sha256(config["path"]),
                                         feature_map=mapper.manifest())
    # Write the complete fixed design before opening data for any outcomes.
    write_json(out / "frozen_protocol.json", manifest)
    metric_rows, selection_rows, correction_rows, score_rows, mechanism_rows = [], [], [], [], []
    data_info = {}
    with threadpool_limits(limits=1):
        for name in args.datasets:
            config = DATASETS[name]
            print(f"Loading {name}; final holdout excluded", flush=True)
            frame = pd.read_csv(config["path"], low_memory=False)
            raw_n = len(frame)
            frame = frame.drop_duplicates(keep="first").reset_index(drop=True)
            train, reserved_test = _safe_train_test_split(frame, config["label"], config["aux"], 0.2, 42)
            data_info[name] = dict(raw_rows=raw_n, deduplicated_rows=len(frame),
                                   development_rows=len(train), reserved_test_rows=len(reserved_test),
                                   final_holdout_transformed=False, final_holdout_evaluated=False)
            del frame, reserved_test
            mapper = FixedFeatureMap(config["numeric"], config["categorical"], protocol.categorical_buckets)
            x = mapper.transform(train)
            y = train[config["label"]].to_numpy(dtype=int)
            attack_type = train[config["aux"]].astype(str).to_numpy()
            del train
            for split in split_seeds:
                partition = _partition_noniid(x, y, attack_type, 4, 0.4, split)
                fits, vals = [], []
                for i, idx in enumerate(partition):
                    strata = np.array([f"{label}::{typ}" for label, typ in zip(y[idx], attack_type[idx])])
                    if pd.Series(strata).value_counts().min() < 2:
                        strata = y[idx]
                    fit, val = train_test_split(idx, test_size=0.2, random_state=split + i, stratify=strata)
                    if len(np.unique(y[fit])) != 2 or len(np.unique(y[val])) != 2:
                        raise ValueError("A client split does not contain both public classes")
                    fits.append(fit)
                    vals.append(val)
                # Check that no validation row enters any released structure.
                all_fit, all_val = np.concatenate(fits), np.concatenate(vals)
                if len(np.intersect1d(all_fit, all_val)) or len(all_fit) + len(all_val) != len(y):
                    raise AssertionError("Invalid fit/validation partition")
                clean = [empirical_structure(x[idx], y[idx], protocol.clip_norm) for idx in fits]
                sizes = np.array([len(idx) for idx in fits])
                w = sizes / sizes.sum()
                sigmas = np.array([protocol.sigma(int(n)) for n in sizes])
                reference = recover(clean, w, np.zeros(4))
                xpool, ypool = x[all_val], y[all_val]
                eval_cache: dict = {}
                for noise in noise_seeds:
                    noisy = [hi + symmetric_noise(np.random.default_rng(np.random.SeedSequence([split, noise, i])),
                                                  hi.shape[0], sigma)
                             for i, (hi, sigma) in enumerate(zip(clean, sigmas))]
                    subsets, full, raw = variant_subsets(noisy, w, sigmas)
                    subsets["Empirical-reference"] = reference["selected"]
                    subsets["All-features"] = [np.arange(x.shape[1])] * 4
                    difference = raw["operator"] - full["operator"]
                    shift = np.trace(difference) / difference.shape[0]
                    projerr = np.linalg.norm(raw["p"] @ raw["p"].T - full["p"] @ full["p"].T, "fro")
                    invariant = all(np.array_equal(a, b) for a, b in zip(full["selected"], raw["selected"]))
                    scalar_error = np.linalg.norm(difference - shift * np.eye(len(difference)), "fro")
                    if not invariant or projerr > 1e-8 or scalar_error > 1e-8:
                        raise AssertionError("Unexpected scalar-correction downstream difference")
                    correction_rows.append(dict(dataset=name, split_seed=split, noise_seed=noise,
                                                projector_error=float(projerr), scalar_identity_error=float(scalar_error),
                                                scalar_shift=float(shift), all_selected_equal=invariant))
                    score_rows.extend(score_diagnostics(full, reference, noisy, clean, name, split, noise))
                    for i, (fit, val) in enumerate(zip(fits, vals)):
                        hp = full["personal"][i][1]
                        shared_q = np.max(np.abs(full["hs"][:-1, -1]))
                        personal_q = np.max(np.abs(hp[:-1, -1]))
                        mechanism_rows.append(dict(
                            dataset=name, split_seed=split, noise_seed=noise, client=i,
                            fit_n=len(fit), validation_n=len(val), sigma=float(sigmas[i]),
                            fit_attack_rate=float(y[fit].mean()), validation_attack_rate=float(y[val].mean()),
                            shared_q_max=float(shared_q), personal_q_max=float(personal_q),
                            retained_personal_rank=int(np.count_nonzero(np.abs(np.linalg.eigvalsh(hp)) > 1e-9)),
                            personal_changes_subset=bool(set(subsets["Full"][i]) != set(subsets["Shared-only"][i])),
                            redundancy_changes_subset=bool(set(subsets["Full"][i]) != set(subsets["No-redundancy"][i])),
                            disagreement_changes_subset=bool(set(subsets["Full"][i]) != set(subsets["No-disagreement"][i])),
                            empirical_sequence_equal=bool(np.array_equal(subsets["Full"][i], reference["selected"][i])),
                            empirical_set_jaccard=len(set(subsets["Full"][i]) & set(reference["selected"][i])) /
                            len(set(subsets["Full"][i]) | set(reference["selected"][i]))))
                        for variant, selected in subsets.items():
                            subset = selected[i]
                            # Order does not change a logistic model. Reuse exact subsets, never outcomes for selection.
                            key = (i, tuple(sorted(subset.tolist())))
                            if key not in eval_cache:
                                eval_cache[key] = evaluate(x[fit], y[fit], x[val], y[val], xpool, ypool, subset)
                            common = dict(dataset=name, split_seed=split, noise_seed=noise, client=i, variant=variant)
                            for view, values in eval_cache[key].items():
                                metric_rows.append(dict(**common, view=view, **values))
                            selection_rows.append(dict(**common, selected=json.dumps(subset.tolist()),
                                                       feature_names=json.dumps([mapper.feature_names[j] for j in subset]),
                                                       selected_raw_field_count=len(set(mapper.feature_names[j].split("::")[1]
                                                                                        for j in subset))))
                    print(f"{name} split={split} noise={noise} complete; correction selections equal={invariant}", flush=True)
                # Incremental artifacts permit auditing even if a later split fails.
                aggregate(out, metric_rows, selection_rows, correction_rows, score_rows, mechanism_rows)
                print(f"{name} split={split} saved; elapsed={time.perf_counter()-started:.1f}s", flush=True)
            del x, y, attack_type
    summary, paired = aggregate(out, metric_rows, selection_rows, correction_rows, score_rows, mechanism_rows)
    report = dict(status="complete", seconds=time.perf_counter()-started, data=data_info,
                  client_metric_rows=len(metric_rows), correction_cases=len(correction_rows),
                  score_steps=len(score_rows), expected_metric_rows=len(args.datasets)*len(split_seeds)*len(noise_seeds)*4*len(VARIANTS)*2,
                  final_holdout_evaluated=False,
                  score_bound_violations=0, correction_selection_differences=0)
    if report["client_metric_rows"] != report["expected_metric_rows"]:
        raise AssertionError("Unexpected number of metric records")
    local = summary[(summary.view == "local") & (summary.metric == "macro_f1")]
    lines = ["# Fixed-protocol training-only pilot", "", "Status: COMPLETE", "",
             "No final holdout was transformed or evaluated. No parameters were tuned after observing results.",
             "All classifier fitting and empirical diagnostics are nonprivate trusted evaluations.", "",
             "## Local validation Macro-F1", "", "| Dataset | Variant | Mean | SD across partitions |",
             "|---|---|---:|---:|"]
    for row in local.itertuples():
        lines.append(f"| {row.dataset} | {row.variant} | {row.mean:.6f} | {row.sd:.6f} |")
    lines += ["", "## Paired differences (Full minus comparator)", "",
              "Exploratory 95% t intervals across partition means; clients/noise are not independent repeats.",
              "Partitions reuse underlying data. Intervals do not establish generalization to new datasets.", "",
              "| Dataset | Comparator | Difference | Low | High | Positive partitions |", "|---|---|---:|---:|---:|---:|"]
    for row in paired[(paired.view == "local") & (paired.metric == "macro_f1")].itertuples():
        low = f"{row.ci95_low:.6f}" if row.ci95_low is not None else "NA"
        high = f"{row.ci95_high:.6f}" if row.ci95_high is not None else "NA"
        lines.append(f"| {row.dataset} | {row.comparator} | {row.full_minus_comparator:.6f} | "
                     f"{low} | {high} | {row.positive_splits}/{row.partition_repeats} |")
    corr, score, mech = pd.DataFrame(correction_rows), pd.DataFrame(score_rows), pd.DataFrame(mechanism_rows)
    lines += ["", "## Mechanism checks", "",
              f"- Scalar correction: {len(corr)} cases; all selectors identical; max projector error {corr.projector_error.max():.3g}.",
              f"- Score inequality: {len(score)} common-prefix steps; zero violations; certified steps {int(score.certified.sum())}.",
              "- A score bound that never certifies a step is valid but cannot support a useful stability guarantee.", ""]
    for dataset, group in mech.groupby("dataset"):
        lines += [f"### {dataset}", "",
                  f"- Personal module changes subset: {group.personal_changes_subset.mean():.1%}.",
                  f"- Redundancy changes subset: {group.redundancy_changes_subset.mean():.1%}.",
                  f"- Disagreement penalty changes subset: {group.disagreement_changes_subset.mean():.1%}.",
                  f"- Mean retained personal rank: {group.retained_personal_rank.mean():.3f}.",
                  f"- Empirical reference set Jaccard: {group.empirical_set_jaccard.mean():.4f}.", ""]
    lines += ["## Boundaries", "",
              "- The epsilon=1 claim covers one structure release per disjoint fixed client only.",
              "- Benchmark deduplication, label-stratified partition construction and their metadata are outside the DP adjacency.",
              "- Repeated seeds are trusted Monte Carlo simulations, not a composed epsilon=1 deployment transcript.",
              "- Public sizes/weights and preprocessing constants are assumed fixed before protected fitting records.",
              "- K=10 counts encoded coordinates; it is not a comparison at 10 raw fields or at the old learned-map dimension.",
              "- This pilot does not provide temporal/device generalization, attack-family robustness or a full theoretical theorem.",
              "- Unbiased scalar correction must not be credited with an effect on fixed-rank selection.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "run_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
