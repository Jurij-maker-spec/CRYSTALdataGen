#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import re
from collections import defaultdict
from math import sqrt
from scipy.stats import beta

import h5py
import numpy as np

REF_PATH = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5")





# Statistics 
def beta_credible_interval(successes, failures, alpha_prior=1.0, beta_prior=1.0, ci=0.90):
    """
    Return lower/upper equal-tailed credible interval for Beta posterior.

    Posterior:
        p ~ Beta(alpha_prior + successes, beta_prior + failures)
    """
    a = alpha_prior + successes
    b = beta_prior + failures

    q_low = (1.0 - ci) / 2.0
    q_high = 1.0 - q_low

    return float(beta.ppf(q_low, a, b)), float(beta.ppf(q_high, a, b))


def make_success_threshold(rows, metric="composite_score", success_quantile=0.10):
    """
    Lower metric is assumed better.
    Success means:
        metric <= quantile(metric, success_quantile)
    """
    values = [
        safe_float(r.get(metric))
        for r in rows
        if np.isfinite(safe_float(r.get(metric)))
    ]

    if not values:
        raise ValueError(f"No finite values found for metric: {metric}")

    threshold = float(np.quantile(values, success_quantile))
    return threshold


def summarize_success_group(
    group_rows,
    *,
    metric,
    threshold,
    alpha_prior=1.0,
    beta_prior=1.0,
    ci=0.90,
):
    scores = [
        safe_float(r.get(metric))
        for r in group_rows
        if np.isfinite(safe_float(r.get(metric)))
    ]

    n = len(scores)
    n_success = sum(score <= threshold for score in scores)
    n_fail = n - n_success

    posterior_mean = (n_success + alpha_prior) / (
        n + alpha_prior + beta_prior
    )

    ci_low, ci_high = beta_credible_interval(
        n_success,
        n_fail,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        ci=ci,
    )

    return {
        "n": n,
        "successes": n_success,
        "failures": n_fail,
        "raw_rate": n_success / n if n > 0 else np.nan,
        "posterior_mean": posterior_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "median_score": median_or_nan(scores),
        "mean_score": mean_or_nan(scores),
        "best_score": min_or_nan(scores),
    }


def probability_best_from_beta(records, n_samples=20000, seed=123):
    """
    Estimate P(each group has the highest success probability).

    records must contain:
        successes
        failures
        label
    """
    rng = np.random.default_rng(seed)

    if not records:
        return {}

    samples = []

    for r in records:
        a = 1.0 + r["successes"]
        b = 1.0 + r["failures"]
        samples.append(rng.beta(a, b, size=n_samples))

    samples = np.vstack(samples)
    winners = np.argmax(samples, axis=0)

    p_best = {}

    for i, r in enumerate(records):
        p_best[r["label"]] = float(np.mean(winners == i))

    return p_best


def print_probabilistic_hyperparameter_analysis(
    rows,
    *,
    metric="composite_score",
    success_quantile=0.10,
    ci=0.90,
    n_samples=20000,
):
    """
    Estimate which hyperparameters are most likely to produce top-performing models.

    Success definition:
        metric <= empirical success_quantile

    Uses Beta-binomial smoothing:
        p(success) ~ Beta(1 + successes, 1 + failures)
    """

    clean_rows = [
        r for r in rows
        if np.isfinite(safe_float(r.get(metric)))
    ]

    if not clean_rows:
        print()
        print(f"No rows available for probabilistic analysis with metric: {metric}")
        return

    threshold = make_success_threshold(
        clean_rows,
        metric=metric,
        success_quantile=success_quantile,
    )

    print()
    print("=" * 120)
    print("PROBABILISTIC HYPERPARAMETER SUCCESS ANALYSIS")
    print("=" * 120)
    print(f"metric             : {metric}")
    print(f"lower is better    : yes")
    print(f"success quantile   : {success_quantile:.3f}")
    print(f"success threshold  : {threshold:.6g}")
    print(f"credible interval  : {100 * ci:.1f}%")
    print()

    pooled_params = [
        ("r_max", "r_max"),
        ("energy_weight", "ew"),
        ("forces_weight", "fw"),
    ]

    for param, short_name in pooled_params:
        groups = {}

        for r in clean_rows:
            value = safe_float(r.get(param))
            if not np.isfinite(value):
                continue

            groups.setdefault(value, []).append(r)

        if not groups:
            continue

        records = []

        for value, group_rows in groups.items():
            summary = summarize_success_group(
                group_rows,
                metric=metric,
                threshold=threshold,
                ci=ci,
            )

            record = {
                "label": value,
                "param": param,
                "value": value,
                **summary,
            }
            records.append(record)

        p_best = probability_best_from_beta(
            records,
            n_samples=n_samples,
            seed=123,
        )

        for r in records:
            r["p_best"] = p_best.get(r["label"], np.nan)

        records = sorted(
            records,
            key=lambda r: (
                -r["posterior_mean"],
                -r["p_best"],
                r["median_score"],
            ),
        )

        print()
        print("-" * 120)
        print(f"POOLED PARAMETER: {param}")
        print("-" * 120)

        header = (
            f"{short_name:>10} "
            f"{'n':>5} "
            f"{'succ':>6} "
            f"{'raw':>8} "
            f"{'post_mean':>10} "
            f"{'CI_low':>10} "
            f"{'CI_high':>10} "
            f"{'P(best)':>10} "
            f"{'median':>10} "
            f"{'best':>10}"
        )
        print(header)
        print("-" * len(header))

        for r in records:
            print(
                f"{r['value']:>10.4g} "
                f"{r['n']:>5d} "
                f"{r['successes']:>6d} "
                f"{100 * r['raw_rate']:>7.1f}% "
                f"{100 * r['posterior_mean']:>9.1f}% "
                f"{100 * r['ci_low']:>9.1f}% "
                f"{100 * r['ci_high']:>9.1f}% "
                f"{100 * r['p_best']:>9.1f}% "
                f"{r['median_score']:>10.4g} "
                f"{r['best_score']:>10.4g}"
            )

    print()
    print("-" * 120)
    print("FULL TUPLE PROBABILITIES")
    print("-" * 120)

    tuple_groups = {}

    for r in clean_rows:
        r_max = safe_float(r.get("r_max"))
        ew = safe_float(r.get("energy_weight"))
        fw = safe_float(r.get("forces_weight"))

        if not all(np.isfinite(v) for v in [r_max, ew, fw]):
            continue

        key = (r_max, ew, fw)
        tuple_groups.setdefault(key, []).append(r)

    tuple_records = []

    for key, group_rows in tuple_groups.items():
        summary = summarize_success_group(
            group_rows,
            metric=metric,
            threshold=threshold,
            ci=ci,
        )

        r_max, ew, fw = key

        tuple_records.append({
            "label": key,
            "r_max": r_max,
            "energy_weight": ew,
            "forces_weight": fw,
            **summary,
        })

    p_best_tuple = probability_best_from_beta(
        tuple_records,
        n_samples=n_samples,
        seed=456,
    )

    for r in tuple_records:
        r["p_best"] = p_best_tuple.get(r["label"], np.nan)

    tuple_records = sorted(
        tuple_records,
        key=lambda r: (
            -r["posterior_mean"],
            -r["p_best"],
            r["median_score"],
        ),
    )

    header = (
        f"{'r_max':>7} "
        f"{'ew':>7} "
        f"{'fw':>7} "
        f"{'n':>5} "
        f"{'succ':>6} "
        f"{'raw':>8} "
        f"{'post_mean':>10} "
        f"{'CI_low':>10} "
        f"{'CI_high':>10} "
        f"{'P(best)':>10} "
        f"{'median':>10} "
        f"{'best':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in tuple_records[:20]:
        print(
            f"{r['r_max']:>7.3g} "
            f"{r['energy_weight']:>7.3g} "
            f"{r['forces_weight']:>7.3g} "
            f"{r['n']:>5d} "
            f"{r['successes']:>6d} "
            f"{100 * r['raw_rate']:>7.1f}% "
            f"{100 * r['posterior_mean']:>9.1f}% "
            f"{100 * r['ci_low']:>9.1f}% "
            f"{100 * r['ci_high']:>9.1f}% "
            f"{100 * r['p_best']:>9.1f}% "
            f"{r['median_score']:>10.4g} "
            f"{r['best_score']:>10.4g}"
        )


def parse_float_token(value: str) -> float:
    value = value.replace("p", ".")
    return float(value)


def parse_split_train_size(split: str, structure: str) -> dict:
    """
    Examples:
      SiO2_1000_10_90      -> train_size_split=1000
      SiO2_PBE_1000_10_90  -> train_size_split=1000
      SiO2_10_90           -> no train size
      SiO2_PBE_sc          -> no train size
    """
    out = {}

    prefix = structure + "_"

    if not split.startswith(prefix):
        return out

    tail = split[len(prefix):]
    parts = tail.split("_")

    if len(parts) >= 3 and parts[0].isdigit():
        out["train_size_split"] = int(parts[0])

    return out


def parse_run_id(run_id: str) -> dict:
    """
    Extract common training parameters from run names.

    Supports examples like:
      MACELES_bs2_ep173_ew1_fw100_rmax3_seed1_SiO2
      MACELES_bs2_ep173_ew1_fw100_rmax3.5_seed1_SiO2
      MACELES_bs2_ep173_ew1_fw100_rmax3p5_seed1_SiO2
    """
    patterns = {
        "model_type": r"^(MACELES|MACE)",
        "batch_size": r"(?:^|_)bs([0-9]+)(?:_|$)",
        "max_epochs": r"(?:^|_)ep([0-9]+)(?:_|$)",
        "energy_weight": r"(?:^|_)ew([0-9.eE+p+-]+)(?:_|$)",
        "forces_weight": r"(?:^|_)fw([0-9.eE+p+-]+)(?:_|$)",
        "r_max": r"(?:^|_)rmax([0-9.eE+p+-]+)(?:_|$)",
        "seed": r"(?:^|_)seed([0-9]+)(?:_|$)",
        "train_size": r"_n([0-9]+)(?:_|$)",
    }

    out = {}

    for key, pat in patterns.items():
        m = re.search(pat, run_id)
        if not m:
            continue

        value = m.group(1)

        if key in {"batch_size", "max_epochs", "seed", "train_size"}:
            value = int(value)
        elif key in {"energy_weight", "forces_weight", "r_max"}:
            value = parse_float_token(value)

        out[key] = value

    return out


def split_matches_structure(split: str, structure: str) -> bool:
    """
    Keep functional variants separated.

    structure='SiO2':
        keep   SiO2_05_95
        keep   SiO2_10_90
        reject SiO2_PBE_1000_10_90
        reject SiO2_PBE_sc

    structure='SiO2_PBE':
        keep   SiO2_PBE_1000_10_90
        keep   SiO2_PBE_sc
        reject SiO2_10_90
    """
    if structure.endswith("_PBE"):
        return split == structure or split.startswith(structure + "_")

    pbe_prefix = structure + "_PBE"
    if split == pbe_prefix or split.startswith(pbe_prefix + "_"):
        return False

    return split == structure or split.startswith(structure + "_")


def iter_evaluation_runs(h5, structure: str):
    path = f"structures/{structure}/evaluations"

    if path not in h5:
        return

    root = h5[path]

    for split, split_group in root.items():
        if not isinstance(split_group, h5py.Group):
            continue

        for sweep_id, sweep_group in split_group.items():
            if not isinstance(sweep_group, h5py.Group):
                continue

            for run_id, run_group in sweep_group.items():
                if not isinstance(run_group, h5py.Group):
                    continue

                yield split, sweep_id, run_id, run_group


def collect_metric_attrs(run_group):
    if "ranking_metrics" not in run_group:
        return {}

    return dict(run_group["ranking_metrics"].attrs)


def get_float(row, key, default=np.nan):
    try:
        return float(row[key])
    except Exception:
        return default


def format_latex_number(value, digits=4):
    try:
        value = float(value)
    except Exception:
        return "--"

    if not np.isfinite(value):
        return "--"

    return f"{value:.{digits}g}"


def safe_float(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default

    if not np.isfinite(value):
        return default

    return value


def safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def median_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def std_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1))


def min_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.min(values))


def max_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.max(values))


def print_best_models(rows, n=7, metric="composite_score"):
    rows = [
        r for r in rows
        if metric in r and np.isfinite(get_float(r, metric))
    ]

    if not rows:
        print(f"No rows with metric: {metric}")
        return

    rows = sorted(rows, key=lambda r: get_float(r, metric))[:n]

    print()
    print(f"BEST {n} MODELS BY {metric}")
    print("-" * 100)

    header = (
        f"{'#':>2} "
        f"{'split':<22} "
        f"{'dataset_size':<15}"
        f"{'r_max':>7} "
        f"{'ew':>7} "
        f"{'fw':>7} "
        f"{'seed':>6} "
        f"{'score':>10} "
        f"{'freq_mae':>10} "
        f"{'int_r':>10} "
        f"{'overlap':>10}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows, start=1):
        overlap = get_float(r, "crystal_mode_mean_overlap")
        if not np.isfinite(overlap):
            overlap = get_float(r, "diagonal_overlap_mean")

        print(
            f"{i:>2d} "
            f"{str(r.get('split', '--')):<22.22} "
            f"{str(r.get('train_size', '--')):<15}"
            f"{format_latex_number(r.get('r_max'), 3):>7} "
            f"{format_latex_number(r.get('energy_weight'), 3):>7} "
            f"{format_latex_number(r.get('forces_weight'), 3):>7} "
            f"{format_latex_number(r.get('seed'), 0):>6} "
            f"{format_latex_number(r.get(metric), 5):>10} "
            f"{format_latex_number(r.get('freq_mae_ir_cm1'), 4):>10} "
            f"{format_latex_number(r.get('intensity_pearson_r'), 4):>10} "
            f"{format_latex_number(overlap, 4):>10}"
        )

    print()
    print("LATEX TABLE")
    print("-" * 100)

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{rrrrrrrr}")
    print(r"\hline")
    print(
        r"$r_\mathrm{max}$ & $w_E$ & $w_F$ & seed & "
        r"score & freq. MAE & $r_I$ & overlap \\"
    )
    print(r"\hline")

    for r in rows:
        overlap = get_float(r, "crystal_mode_mean_overlap")
        if not np.isfinite(overlap):
            overlap = get_float(r, "diagonal_overlap_mean")

        print(
            f"{format_latex_number(r.get('r_max'), 3)} & "
            f"{format_latex_number(r.get('energy_weight'), 3)} & "
            f"{format_latex_number(r.get('forces_weight'), 3)} & "
            f"{format_latex_number(r.get('seed'), 0)} & "
            f"{format_latex_number(r.get(metric), 5)} & "
            f"{format_latex_number(r.get('freq_mae_ir_cm1'), 4)} & "
            f"{format_latex_number(r.get('intensity_pearson_r'), 4)} & "
            f"{format_latex_number(overlap, 4)} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Best model hyperparameters and evaluation metrics.}")
    print(r"\label{tab:best_models}")
    print(r"\end{table}")


def print_hyperparameter_ranking(
    rows,
    metric="composite_score",
    n_best=20,
):
    """
    Rank hyperparameter combinations by median metric.

    Hyperparameters screened:
      r_max, energy_weight, forces_weight

    train_size is not used for grouping yet.
    It is only printed as additional information if available.
    """
    required = ["r_max", "energy_weight", "forces_weight", metric]

    clean_rows = []
    for r in rows:
        if not all(k in r for k in required):
            continue

        score = safe_float(r.get(metric))
        if not np.isfinite(score):
            continue

        clean_rows.append(r)

    if not clean_rows:
        print()
        print(f"No rows available for hyperparameter ranking with metric: {metric}")
        return

    groups = {}

    for r in clean_rows:
        key = (
            safe_float(r["r_max"]),
            safe_float(r["energy_weight"]),
            safe_float(r["forces_weight"]),
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(r)

    records = []

    global_best = min_or_nan([safe_float(r.get(metric)) for r in clean_rows])
    succes_multiplier = 2.3
    success_threshold = succes_multiplier * global_best

    for (r_max, ew, fw), group_rows in groups.items():
        scores = [safe_float(r.get(metric)) for r in group_rows]

        freq_mae = [safe_float(r.get("freq_mae_ir_cm1")) for r in group_rows]
        spec_l2 = [safe_float(r.get("spectrum_rel_l2")) for r in group_rows]
        int_r = [safe_float(r.get("intensity_pearson_r")) for r in group_rows]

        overlap_values = []
        for r in group_rows:
            overlap = safe_float(r.get("crystal_mode_mean_overlap"))
            if not np.isfinite(overlap):
                overlap = safe_float(r.get("diagonal_overlap_mean"))
            if np.isfinite(overlap):
                overlap_values.append(overlap)

        train_sizes = []
        for r in group_rows:
            ts = safe_int(r.get("train_size"))
            if ts is not None:
                train_sizes.append(ts)

        splits = sorted(set(str(r.get("split", "--")) for r in group_rows))
        seeds = sorted(set(safe_int(r.get("seed")) for r in group_rows if safe_int(r.get("seed")) is not None))

        n_success = sum(
            safe_float(r.get(metric)) <= success_threshold
            for r in group_rows
        )

        record = {
            "r_max": r_max,
            "energy_weight": ew,
            "forces_weight": fw,
            "n_runs": len(group_rows),
            "n_splits": len(splits),
            "n_seeds": len(seeds),
            "train_size_min": min(train_sizes) if train_sizes else None,
            "train_size_max": max(train_sizes) if train_sizes else None,
            "median_score": median_or_nan(scores),
            "mean_score": mean_or_nan(scores),
            "std_score": std_or_nan(scores),
            "best_score": min_or_nan(scores),
            "worst_score": max_or_nan(scores),
            "success_rate": n_success / len(group_rows),
            "median_freq_mae": median_or_nan(freq_mae),
            "median_spectrum_l2": median_or_nan(spec_l2),
            "median_intensity_r": median_or_nan(int_r),
            "median_overlap": median_or_nan(overlap_values),
        }

        records.append(record)

    records = sorted(
        records,
        key=lambda r: (
            r["median_score"],
            r["mean_score"],
            r["std_score"] if np.isfinite(r["std_score"]) else np.inf,
        ),
    )

    print()
    print("=" * 130)
    print(f"HYPERPARAMETER RANKING BY MEDIAN {metric}")
    print("=" * 130)
    print(f"global best score : {global_best:.6g}")
    print(f"success threshold : {success_threshold:.6g}  (= {succes_multiplier} * global best)")
    print()

    header = (
        f"{'#':>3} "
        f"{'r_max':>7} "
        f"{'ew':>7} "
        f"{'fw':>7} "
        f"{'n':>5} "
        f"{'splits':>7} "
        f"{'seeds':>6} "
        f"{'Ntrain':>15} "
        f"{'median':>10} "
        f"{'mean':>10} "
        f"{'std':>10} "
        f"{'best':>10} "
        f"{'succ.':>8} "
        f"{'freqMAE':>10} "
        f"{'specL2':>10} "
        f"{'intR':>8} "
        f"{'overlap':>9}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(records[:n_best], start=1):
        if r["train_size_min"] is None:
            train_size_text = "--"
        elif r["train_size_min"] == r["train_size_max"]:
            train_size_text = f"{r['train_size_min']}"
        else:
            train_size_text = f"{r['train_size_min']}-{r['train_size_max']}"

        print(
            f"{i:>3d} "
            f"{r['r_max']:>7.3g} "
            f"{r['energy_weight']:>7.3g} "
            f"{r['forces_weight']:>7.3g} "
            f"{r['n_runs']:>5d} "
            f"{r['n_splits']:>7d} "
            f"{r['n_seeds']:>6d} "
            f"{train_size_text:>15} "
            f"{r['median_score']:>10.4g} "
            f"{r['mean_score']:>10.4g} "
            f"{r['std_score']:>10.4g} "
            f"{r['best_score']:>10.4g} "
            f"{100.0 * r['success_rate']:>7.1f}% "
            f"{r['median_freq_mae']:>10.4g} "
            f"{r['median_spectrum_l2']:>10.4g} "
            f"{r['median_intensity_r']:>8.3g} "
            f"{r['median_overlap']:>9.3g}"
        )

    print()
    print("LATEX TABLE")
    print("-" * 130)

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{rrrrrrrrrrr}")
    print(r"\hline")
    print(
        r"$r_\mathrm{max}$ & $w_E$ & $w_F$ & $n$ & $N_\mathrm{train}$ & "
        r"median & mean & std & best & freq. MAE & $r_I$ \\"
    )
    print(r"\hline")

    for r in records[:n_best]:
        if r["train_size_min"] is None:
            train_size_text = "--"
        elif r["train_size_min"] == r["train_size_max"]:
            train_size_text = str(r["train_size_min"])
        else:
            train_size_text = f"{r['train_size_min']}--{r['train_size_max']}"

        print(
            f"{format_latex_number(r['r_max'], 3)} & "
            f"{format_latex_number(r['energy_weight'], 3)} & "
            f"{format_latex_number(r['forces_weight'], 3)} & "
            f"{r['n_runs']} & "
            f"{train_size_text} & "
            f"{format_latex_number(r['median_score'], 4)} & "
            f"{format_latex_number(r['mean_score'], 4)} & "
            f"{format_latex_number(r['std_score'], 3)} & "
            f"{format_latex_number(r['best_score'], 4)} & "
            f"{format_latex_number(r['median_freq_mae'], 4)} & "
            f"{format_latex_number(r['median_intensity_r'], 3)} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(
        r"\caption{Hyperparameter ranking grouped by cutoff radius, "
        r"energy weight, and force weight. Scores are aggregated over "
        r"available seeds, splits, sweeps, and training-set sizes.}"
    )
    print(r"\label{tab:hyperparameter_ranking}")
    print(r"\end{table}")


def print_pooled_hyperparameter_ranking(
    rows,
    metric="composite_score",
):
    """
    Main-effect / pooled hyperparameter ranking.

    For each parameter separately:
      r_max
      energy_weight
      forces_weight

    Pool over all other parameters, seeds, splits, sweeps, and train sizes.
    """

    parameters = [
        ("r_max", r"$r_\mathrm{max}$"),
        ("energy_weight", r"$w_E$"),
        ("forces_weight", r"$w_F$"),
    ]

    clean_rows = []
    for r in rows:
        score = safe_float(r.get(metric))
        if not np.isfinite(score):
            continue
        clean_rows.append(r)

    if not clean_rows:
        print()
        print(f"No rows available for pooled hyperparameter ranking with metric: {metric}")
        return

    global_best = min_or_nan([safe_float(r.get(metric)) for r in clean_rows])
    success_multiplier = 2.3
    success_threshold = success_multiplier * global_best

    print()
    print("=" * 120)
    print(f"POOLED HYPERPARAMETER EFFECTS BY {metric}")
    print("=" * 120)
    print(f"global best score : {global_best:.6g}")
    print(f"success threshold : {success_threshold:.6g}  (= {success_multiplier} * global best)")
    print()

    pooled_best = {}

    for param, latex_name in parameters:
        groups = {}

        for r in clean_rows:
            if param not in r:
                continue

            value = safe_float(r.get(param))
            if not np.isfinite(value):
                continue

            groups.setdefault(value, []).append(r)

        if not groups:
            continue

        records = []

        for value, group_rows in groups.items():
            scores = [safe_float(r.get(metric)) for r in group_rows]

            freq_mae = [safe_float(r.get("freq_mae_ir_cm1")) for r in group_rows]
            spec_l2 = [safe_float(r.get("spectrum_rel_l2")) for r in group_rows]
            int_r = [safe_float(r.get("intensity_pearson_r")) for r in group_rows]

            overlap_values = []
            for r in group_rows:
                overlap = safe_float(r.get("crystal_mode_mean_overlap"))
                if not np.isfinite(overlap):
                    overlap = safe_float(r.get("diagonal_overlap_mean"))
                if np.isfinite(overlap):
                    overlap_values.append(overlap)

            train_sizes = []
            for r in group_rows:
                ts = safe_int(r.get("train_size"))
                if ts is not None:
                    train_sizes.append(ts)

            splits = sorted(set(str(r.get("split", "--")) for r in group_rows))
            seeds = sorted(
                set(
                    safe_int(r.get("seed"))
                    for r in group_rows
                    if safe_int(r.get("seed")) is not None
                )
            )

            n_success = sum(
                safe_float(r.get(metric)) <= success_threshold
                for r in group_rows
            )

            record = {
                "param": param,
                "value": value,
                "n_runs": len(group_rows),
                "n_splits": len(splits),
                "n_seeds": len(seeds),
                "train_size_min": min(train_sizes) if train_sizes else None,
                "train_size_max": max(train_sizes) if train_sizes else None,
                "median_score": median_or_nan(scores),
                "mean_score": mean_or_nan(scores),
                "std_score": std_or_nan(scores),
                "best_score": min_or_nan(scores),
                "worst_score": max_or_nan(scores),
                "success_rate": n_success / len(group_rows),
                "median_freq_mae": median_or_nan(freq_mae),
                "median_spectrum_l2": median_or_nan(spec_l2),
                "median_intensity_r": median_or_nan(int_r),
                "median_overlap": median_or_nan(overlap_values),
            }

            records.append(record)

        records = sorted(
            records,
            key=lambda r: (
                r["median_score"],
                r["mean_score"],
                r["std_score"] if np.isfinite(r["std_score"]) else np.inf,
            ),
        )

        pooled_best[param] = records[0]

        print()
        print("-" * 120)
        print(f"POOLED PARAMETER: {param}")
        print("-" * 120)

        header = (
            f"{'value':>10} "
            f"{'n':>5} "
            f"{'splits':>7} "
            f"{'seeds':>6} "
            f"{'Ntrain':>15} "
            f"{'median':>10} "
            f"{'mean':>10} "
            f"{'std':>10} "
            f"{'best':>10} "
            f"{'succ.':>8} "
            f"{'freqMAE':>10} "
            f"{'specL2':>10} "
            f"{'intR':>8} "
            f"{'overlap':>9}"
        )
        print(header)
        print("-" * len(header))

        for r in records:
            if r["train_size_min"] is None:
                train_size_text = "--"
            elif r["train_size_min"] == r["train_size_max"]:
                train_size_text = f"{r['train_size_min']}"
            else:
                train_size_text = f"{r['train_size_min']}-{r['train_size_max']}"

            print(
                f"{r['value']:>10.4g} "
                f"{r['n_runs']:>5d} "
                f"{r['n_splits']:>7d} "
                f"{r['n_seeds']:>6d} "
                f"{train_size_text:>15} "
                f"{r['median_score']:>10.4g} "
                f"{r['mean_score']:>10.4g} "
                f"{r['std_score']:>10.4g} "
                f"{r['best_score']:>10.4g} "
                f"{100.0 * r['success_rate']:>7.1f}% "
                f"{r['median_freq_mae']:>10.4g} "
                f"{r['median_spectrum_l2']:>10.4g} "
                f"{r['median_intensity_r']:>8.3g} "
                f"{r['median_overlap']:>9.3g}"
            )

    print()
    print("=" * 120)
    print("BEST POOLED VALUES")
    print("=" * 120)

    for param in ["r_max", "energy_weight", "forces_weight"]:
        if param not in pooled_best:
            continue

        r = pooled_best[param]
        print(
            f"{param:<15} = {r['value']:<8.4g} "
            f"median={r['median_score']:.4g}, "
            f"mean={r['mean_score']:.4g}, "
            f"n={r['n_runs']}"
        )

    if all(k in pooled_best for k in ["r_max", "energy_weight", "forces_weight"]):
        best_tuple = (
            pooled_best["r_max"]["value"],
            pooled_best["energy_weight"]["value"],
            pooled_best["forces_weight"]["value"],
        )

        matching_rows = [
            r for r in clean_rows
            if np.isclose(safe_float(r.get("r_max")), best_tuple[0])
            and np.isclose(safe_float(r.get("energy_weight")), best_tuple[1])
            and np.isclose(safe_float(r.get("forces_weight")), best_tuple[2])
        ]

        print()
        print("OVERLAP CHECK")
        print("-" * 120)
        print(
            f"best pooled tuple: "
            f"r_max={best_tuple[0]:g}, "
            f"ew={best_tuple[1]:g}, "
            f"fw={best_tuple[2]:g}"
        )

        if matching_rows:
            scores = [safe_float(r.get(metric)) for r in matching_rows]
            print(f"existing runs    : {len(matching_rows)}")
            print(f"median score     : {median_or_nan(scores):.6g}")
            print(f"mean score       : {mean_or_nan(scores):.6g}")
            print(f"best score       : {min_or_nan(scores):.6g}")
        else:
            print("existing runs    : none")
            print("interpretation   : best pooled values do not occur as one exact tuple")


def add_if_numeric(store, key, value):
    try:
        value = float(value)
    except Exception:
        return
    if np.isfinite(value):
        store[key].append(value)


def summarize_structure(
    h5,
    structure: str,
    include_all_splits: bool = False,
    rank_hyperparams: bool = False,
    pool_hyperparams: bool = False,
    prob_hyperparams: bool = False,
    rank_metric: str = "composite_score",
    rank_top: int = 20,
    success_quantile: float = 0.10,
    prob_ci: float = 0.90,
    prob_samples: int = 20000,
):
    all_runs = list(iter_evaluation_runs(h5, structure) or [])
    all_runs = list(iter_evaluation_runs(h5, structure) or [])

    if include_all_splits:
        runs = all_runs
    else:
        runs = [
            row for row in all_runs
            if split_matches_structure(row[0], structure)
        ]
    if not runs:
        print(f"No evaluations found for structure: {structure}")
        return

    print(f"model evaluations : {len(runs)}")
    print(f"filtered out       : {len(all_runs) - len(runs)}")

    param_values = defaultdict(set)
    metric_values = defaultdict(list)

    split_values = set()
    sweep_values = set()

    missing_param_counts = defaultdict(int)

    model_rows = []
    for split, sweep_id, run_id, run_group in runs:
        split_values.add(split)
        sweep_values.add((split, sweep_id))

        params = parse_run_id(run_id)
        split_info = parse_split_train_size(split, structure)

        train_size = params.get("train_size")
        if train_size is None:
            train_size = split_info.get("train_size_split")

        for key, value in params.items():
            param_values[key].add(value)

        for expected in [
            "model_type",
            "batch_size",
            "max_epochs",
            "energy_weight",
            "forces_weight",
            "r_max",
            "seed",
        ]:
            if expected not in params:
                missing_param_counts[expected] += 1

        metrics = collect_metric_attrs(run_group)
        row = {
            "split": split,
            "sweep_id": sweep_id,
            "run_id": run_id,
            **split_info,
            **params,
            "train_size": train_size,
        }


        for key, value in metrics.items():
            try:
                row[key] = float(value)
            except Exception:
                row[key] = value

        if "imag_flags" in run_group:
            row["n_imag_model"] = int(np.sum(run_group["imag_flags"][()]))

        model_rows.append(row)
        for key, value in metrics.items():
            add_if_numeric(metric_values, key, value)

        if "imag_flags" in run_group:
            n_imag = int(np.sum(run_group["imag_flags"][()]))
            metric_values["n_imag_model"].append(float(n_imag))

    print("=" * 100)
    print(f"REF DB SWEEP EXPLORER: {structure}")
    print("=" * 100)
    print(f"model evaluations : {len(runs)}")
    print(f"splits            : {len(split_values)}")
    print(f"sweeps            : {len(sweep_values)}")
    print()

    print("SPLITS")
    print("-" * 100)
    for split in sorted(split_values):
        n = sum(1 for s, _, _, _ in runs if s == split)
        print(f"{split:<30} {n:>6} runs")
    print()

    print("TRAINING PARAMETERS PARSED FROM run_id")
    print("-" * 100)
    header = f"{'parameter':<20} {'n_unique':>10}  values"
    print(header)
    print("-" * len(header))

    for key in sorted(param_values):
        values = sorted(param_values[key], key=lambda x: str(x))
        print(f"{key:<20} {len(values):>10}  {values}")

    print()
    print("MISSING PARAMETER COUNTS")
    print("-" * 100)
    for key in sorted(missing_param_counts):
        count = missing_param_counts[key]
        if count > 0:
            print(f"{key:<20} {count:>10} / {len(runs)}")
    print()

    print("METRICS FOUND IN ranking_metrics")
    print("-" * 100)
    header = (
        f"{'metric':<35} "
        f"{'n':>6} "
        f"{'min':>14} "
        f"{'max':>14} "
        f"{'mean':>14}"
    )
    print(header)
    print("-" * len(header))

    for key in sorted(metric_values):
        values = np.asarray(metric_values[key], dtype=float)
        print(
            f"{key:<35} "
            f"{len(values):>6d} "
            f"{np.nanmin(values):>14.6g} "
            f"{np.nanmax(values):>14.6g} "
            f"{np.nanmean(values):>14.6g}"
        )

    print_best_models(model_rows, n=7, metric="composite_score")
    if rank_hyperparams:
        print_hyperparameter_ranking(
            model_rows,
            metric=rank_metric,
            n_best=rank_top,
        )
    if pool_hyperparams:
        print_pooled_hyperparameter_ranking(
            model_rows,
            metric=rank_metric,
        )
    if prob_hyperparams:
        print_probabilistic_hyperparameter_analysis(
            model_rows,
            metric=rank_metric,
            success_quantile=success_quantile,
            ci=prob_ci,
            n_samples=prob_samples,
        )  


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore available sweep parameters and metrics in ref_db.h5."
    )
    parser.add_argument("--ref_db", type=Path, default=REF_PATH)
    parser.add_argument("--structure", required=True)

    parser.add_argument(
        "--include_all_splits",
        action="store_true",
        help="Include all splits under the structure group, including e.g. SiO2_PBE splits under SiO2.",
    )

    parser.add_argument(
        "--rank-hyperparams",
        action="store_true",
        help="Rank r_max, energy_weight, forces_weight combinations by median score.",
    )
    parser.add_argument(
        "--rank-metric",
        default="composite_score",
        help="Metric used for hyperparameter ranking.",
    )
    parser.add_argument(
        "--rank-top",
        type=int,
        default=20,
        help="Number of hyperparameter combinations to print.",
    )
    parser.add_argument(
        "--pool-hyperparams",
        action="store_true",
        help="Print pooled main-effect ranking for r_max, energy_weight, and forces_weight.",
    )
    parser.add_argument(
        "--prob_hyperparams",
        action="store_true",
        help="Estimate probability that hyperparameters yield top-performing models.",
    )

    parser.add_argument(
        "--success_quantile",
        type=float,
        default=0.10,
        help="Success threshold as lower quantile of the ranking metric. Default: 0.10 means best 10%%.",
    )

    parser.add_argument(
        "--prob_ci",
        type=float,
        default=0.90,
        help="Credible interval width for Beta-binomial probabilities.",
    )

    parser.add_argument(
        "--prob_samples",
        type=int,
        default=20000,
        help="Monte Carlo samples for estimating P(best).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with h5py.File(args.ref_db, "r") as h5:
        if "structures" not in h5:
            raise KeyError("Missing top-level group: /structures")

        if args.structure not in h5["structures"]:
            raise KeyError(f"Structure not found: {args.structure}")

        summarize_structure(
            h5,
            args.structure,
            include_all_splits=args.include_all_splits,
            rank_hyperparams=args.rank_hyperparams,
            pool_hyperparams=args.pool_hyperparams,
            prob_hyperparams=args.prob_hyperparams,
            rank_metric=args.rank_metric,
            rank_top=args.rank_top,
            success_quantile=args.success_quantile,
            prob_ci=args.prob_ci,
            prob_samples=args.prob_samples,
        )


if __name__ == "__main__":
    main()

