#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import re
from collections import defaultdict
from scipy.stats import beta

import h5py
import numpy as np

REF_PATH = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5")


# ============================================================
# Generic conversion / safe statistics
# ============================================================

def decode_hdf5_scalar(value):
    """Convert common HDF5 scalar values to normal Python types."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value
    return value


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
        if value is None:
            return default
        value_f = float(value)
        if not np.isfinite(value_f):
            return default
        return int(round(value_f))
    except Exception:
        return default


def get_float(row, key, default=np.nan):
    return safe_float(row.get(key), default=default)


def format_latex_number(value, digits=4):
    value = safe_float(value)
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}g}"


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


def add_if_numeric(store, key, value):
    value = safe_float(value)
    if np.isfinite(value):
        store[key].append(value)


# ============================================================
# DB / path / run_id parsing
# ============================================================

def parse_float_token(value: str) -> float:
    return float(value.replace("p", "."))


def parse_run_id(run_id: str) -> dict:
    """
    Fallback parser for common training parameters from run names.

    DB hyperparameters are preferred. This is only fallback/diagnostic.
    """
    patterns = {
        "model_type": r"^(MACELES|MACE)",
        "batch_size": r"(?:^|_)bs([0-9]+)(?:_|$)",
        "max_epochs": r"(?:^|_)ep([0-9]+)(?:_|$)",
        "energy_weight": r"(?:^|_)ew([0-9.eE+p+-]+)(?:_|$)",
        "forces_weight": r"(?:^|_)fw([0-9.eE+p+-]+)(?:_|$)",
        "r_max": r"(?:^|_)rmax([0-9.eE+p+-]+)(?:_|$)",
        "seed": r"(?:^|_)seed([0-9]+)(?:_|$)",
        # Old cumulative-size suffix. Store as parsed size, not train_size.
        "size": r"_n([0-9]+)(?:_|$)",
    }

    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, run_id)
        if not m:
            continue
        value = m.group(1)
        if key in {"batch_size", "max_epochs", "seed", "size"}:
            value = int(value)
        elif key in {"energy_weight", "forces_weight", "r_max"}:
            value = parse_float_token(value)
        out[key] = value
    return out


def parse_split_size(path_split: str, structure: str) -> dict:
    """
    Parse possible size-like information from the path-level split name.

    This is fallback/diagnostic only. The path-level split is not authoritative.
    Authoritative size and split_val should come from DB hyperparameters.
    """
    out = {}
    prefix = structure + "_"
    if not path_split.startswith(prefix):
        return out

    tail = path_split[len(prefix):]
    parts = tail.split("_")

    if len(parts) >= 3 and parts[0].isdigit():
        out["size_split"] = int(parts[0])

    return out


def split_matches_structure(path_split: str, structure: str) -> bool:
    """
    Keep functional variants separated when scanning path-level splits.
    """
    if structure.endswith("_PBE"):
        return path_split == structure or path_split.startswith(structure + "_")

    pbe_prefix = structure + "_PBE"
    if path_split == pbe_prefix or path_split.startswith(pbe_prefix + "_"):
        return False

    return path_split == structure or path_split.startswith(structure + "_")


def iter_evaluation_runs(h5, structure: str):
    path = f"structures/{structure}/evaluations"
    if path not in h5:
        return

    root = h5[path]
    for path_split, split_group in root.items():
        if not isinstance(split_group, h5py.Group):
            continue
        for sweep_id, sweep_group in split_group.items():
            if not isinstance(sweep_group, h5py.Group):
                continue
            for run_id, run_group in sweep_group.items():
                if not isinstance(run_group, h5py.Group):
                    continue
                yield path_split, sweep_id, run_id, run_group


def collect_metric_attrs(run_group):
    if "ranking_metrics" not in run_group:
        return {}
    return dict(run_group["ranking_metrics"].attrs)


# ============================================================
# Per-run DB hyperparameters: authoritative source
# ============================================================

def read_hyperparameters_from_run_group(run_group):
    """
    Read per-run hyperparameters from the DB.

    Preferred layout:
        run_group["hyperparameters"].attrs[...]

    Also supports scalar datasets inside /hyperparameters.
    """
    if "hyperparameters" not in run_group:
        return {}

    hp_group = run_group["hyperparameters"]
    out = {}

    for key, value in hp_group.attrs.items():
        out[key] = decode_hdf5_scalar(value)

    for key, item in hp_group.items():
        if isinstance(item, h5py.Dataset):
            try:
                out[key] = decode_hdf5_scalar(item[()])
            except Exception:
                pass

    return out


def normalize_hyperparameter_dict(hp):
    """
    Normalize DB hyperparameter values.

    Current important DB fields:
        size
        split_val
        batch_size
        max_epochs
        energy_weight
        forces_weight
        r_max
        seed
    """
    int_keys = {"batch_size", "max_epochs", "seed", "size"}
    float_keys = {"energy_weight", "forces_weight", "r_max", "split_val"}

    out = {}
    for key, value in hp.items():
        if key in int_keys:
            parsed = safe_int(value)
            out[key] = parsed if parsed is not None else value
        elif key in float_keys:
            parsed = safe_float(value)
            out[key] = parsed if np.isfinite(parsed) else value
        else:
            parsed_float = safe_float(value)
            out[key] = parsed_float if np.isfinite(parsed_float) else value
    return out


def prefix_dict(d, prefix):
    return {f"{prefix}{key}": value for key, value in d.items()}


def resolve_parameter(key, *, db_hyperparams, parsed_params, split_info=None):
    """
    Resolve one parameter using the hierarchy:
        1. DB hyperparameters
        2. parsed run_id parameters
        3. parsed path split info, only for size fallback
    """
    if key in db_hyperparams:
        return db_hyperparams[key], "db"

    if key in parsed_params:
        return parsed_params[key], "run_id"

    if key == "size" and split_info is not None and "size_split" in split_info:
        return split_info["size_split"], "path_split"

    return None, "missing"


def parameter_status(key, *, db_hyperparams, parsed_params, split_info=None):
    values = {}

    if key in db_hyperparams:
        values["db"] = db_hyperparams[key]
    if key in parsed_params:
        values["run_id"] = parsed_params[key]
    if key == "size" and split_info is not None and "size_split" in split_info:
        values["path_split"] = split_info["size_split"]

    if not values:
        return "missing"
    if len(values) == 1:
        return f"{next(iter(values))}_only"

    numeric_values = [safe_float(v) for v in values.values()]
    if all(np.isfinite(v) for v in numeric_values):
        if all(np.isclose(v, numeric_values[0]) for v in numeric_values):
            return "consistent"
        return "mismatch"

    if len(set(str(v) for v in values.values())) == 1:
        return "consistent"
    return "mismatch"


def make_analysis_row(*, structure, path_split, sweep_id, run_id, run_group):
    """
    Build one model-analysis row.

    DB hyperparameters are primary.
    run_id parsing is fallback.
    path_split is traceability metadata only.
    """
    db_hyperparams = normalize_hyperparameter_dict(
        read_hyperparameters_from_run_group(run_group)
    )
    parsed_params = parse_run_id(run_id)
    split_info = parse_split_size(path_split, structure)

    analysis_keys = [
        "model_type",
        "batch_size",
        "max_epochs",
        "energy_weight",
        "forces_weight",
        "r_max",
        "seed",
        "size",
        "split_val",
    ]

    resolved = {}
    sources = {}
    statuses = {}

    for key in analysis_keys:
        value, source = resolve_parameter(
            key,
            db_hyperparams=db_hyperparams,
            parsed_params=parsed_params,
            split_info=split_info,
        )
        resolved[key] = value
        sources[f"{key}_source"] = source
        statuses[f"{key}_status"] = parameter_status(
            key,
            db_hyperparams=db_hyperparams,
            parsed_params=parsed_params,
            split_info=split_info,
        )

    row = {
        "structure": structure,
        "split": path_split,       # backward compatibility
        "path_split": path_split,  # preferred interpretation
        "sweep_id": sweep_id,
        "run_id": run_id,
        **resolved,
        **sources,
        **statuses,
        **prefix_dict(db_hyperparams, "db_"),
        **prefix_dict(parsed_params, "run_id_"),
        **prefix_dict(split_info, "path_"),
    }

    metrics = collect_metric_attrs(run_group)
    for key, value in metrics.items():
        try:
            row[key] = float(value)
        except Exception:
            row[key] = value

    if "imag_flags" in run_group:
        row["n_imag_model"] = int(np.sum(run_group["imag_flags"][()]))

    return row


# ============================================================
# Normalization for combined structures
# ============================================================

def percentile_rank_lower_is_better(values):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    if np.sum(finite) == 0:
        return out

    finite_values = values[finite]
    order = np.argsort(finite_values)
    ranks = np.empty(len(finite_values), dtype=float)
    sorted_values = finite_values[order]

    i = 0
    while i < len(sorted_values):
        j = i + 1
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j

    if len(finite_values) > 1:
        ranks = ranks / (len(finite_values) - 1)
    else:
        ranks[:] = 0.0

    out[finite] = ranks
    return out


def add_normalized_metric(rows, *, metric="composite_score", group_key="structure", method="percentile", out_key=None):
    """Add normalized score to rows. Lower is always better."""
    if out_key is None:
        out_key = f"{metric}_{method}"

    if method == "none":
        for r in rows:
            r[out_key] = safe_float(r.get(metric))
        return out_key

    groups = defaultdict(list)
    for idx, r in enumerate(rows):
        value = safe_float(r.get(metric))
        group = r.get(group_key, "--")
        if np.isfinite(value):
            groups[group].append(idx)

    for r in rows:
        r[out_key] = np.nan

    for group, indices in groups.items():
        values = np.asarray([safe_float(rows[i].get(metric)) for i in indices], dtype=float)

        if method == "percentile":
            norm_values = percentile_rank_lower_is_better(values)
        elif method == "zscore":
            mean = np.nanmean(values)
            std = np.nanstd(values, ddof=1)
            norm_values = values * np.nan if (not np.isfinite(std) or std == 0.0) else (values - mean) / std
        elif method == "relative_best":
            best = np.nanmin(values)
            norm_values = values * np.nan if (not np.isfinite(best) or best == 0.0) else values / best
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        for idx, norm_value in zip(indices, norm_values):
            rows[idx][out_key] = float(norm_value)

    return out_key


# ============================================================
# Probabilistic helper functions
# ============================================================

def beta_credible_interval(successes, failures, alpha_prior=1.0, beta_prior=1.0, ci=0.90):
    a = alpha_prior + successes
    b = beta_prior + failures
    q_low = (1.0 - ci) / 2.0
    q_high = 1.0 - q_low
    return float(beta.ppf(q_low, a, b)), float(beta.ppf(q_high, a, b))


def make_success_threshold(rows, metric="composite_score", success_quantile=0.10):
    values = [safe_float(r.get(metric)) for r in rows if np.isfinite(safe_float(r.get(metric)))]
    if not values:
        raise ValueError(f"No finite values found for metric: {metric}")
    return float(np.quantile(values, success_quantile))


def summarize_success_group(group_rows, *, metric, threshold, alpha_prior=1.0, beta_prior=1.0, ci=0.90):
    scores = [safe_float(r.get(metric)) for r in group_rows if np.isfinite(safe_float(r.get(metric)))]
    n = len(scores)
    n_success = sum(score <= threshold for score in scores)
    n_fail = n - n_success
    posterior_mean = (n_success + alpha_prior) / (n + alpha_prior + beta_prior)
    ci_low, ci_high = beta_credible_interval(n_success, n_fail, alpha_prior=alpha_prior, beta_prior=beta_prior, ci=ci)
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
    return {r["label"]: float(np.mean(winners == i)) for i, r in enumerate(records)}


# ============================================================
# Printing functions
# ============================================================

def print_best_models(rows, n=7, metric="composite_score"):
    rows = [r for r in rows if metric in r and np.isfinite(get_float(r, metric))]
    if not rows:
        print(f"No rows with metric: {metric}")
        return

    rows = sorted(rows, key=lambda r: get_float(r, metric))[:n]

    print()
    print(f"BEST {n} MODELS BY {metric}")
    print("-" * 130)

    header = (
        f"{'#':>2} "
        f"{'structure':<14} "
        f"{'path_split':<22} "
        f"{'size':>8} "
        f"{'split_val':>9} "
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
            f"{str(r.get('structure', '--')):<14.14} "
            f"{str(r.get('path_split', r.get('split', '--'))):<22.22} "
            f"{format_latex_number(r.get('size'), 0):>8} "
            f"{format_latex_number(r.get('split_val'), 3):>9} "
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
    print("-" * 130)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{rrrrrrrrrr}")
    print(r"\hline")
    print(r"$N$ & $s_\mathrm{val}$ & $r_\mathrm{max}$ & $w_E$ & $w_F$ & seed & score & freq. MAE & $r_I$ & overlap \\")
    print(r"\hline")

    for r in rows:
        overlap = get_float(r, "crystal_mode_mean_overlap")
        if not np.isfinite(overlap):
            overlap = get_float(r, "diagonal_overlap_mean")
        print(
            f"{format_latex_number(r.get('size'), 0)} & "
            f"{format_latex_number(r.get('split_val'), 3)} & "
            f"{format_latex_number(r.get('r_max'), 3)} & "
            f"{format_latex_number(r.get('energy_weight'), 3)} & "
            f"{format_latex_number(r.get('forces_weight'), 3)} & "
            f"{format_latex_number(r.get('seed'), 0)} & "
            f"{format_latex_number(r.get(metric), 5)} & "
            f"{format_latex_number(r.get('freq_mae_ir_cm1'), 4)} & "
            f"{format_latex_number(r.get('intensity_pearson_r'), 3)} & "
            f"{format_latex_number(overlap, 4)} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Best model hyperparameters and evaluation metrics.}")
    print(r"\label{tab:best_models}")
    print(r"\end{table}")


def _size_split_text(group_rows):
    sizes = [safe_int(r.get("size")) for r in group_rows if safe_int(r.get("size")) is not None]
    split_vals = [safe_float(r.get("split_val")) for r in group_rows if np.isfinite(safe_float(r.get("split_val")))]

    if not sizes:
        size_text = "--"
    elif min(sizes) == max(sizes):
        size_text = f"{min(sizes)}"
    else:
        size_text = f"{min(sizes)}-{max(sizes)}"

    if not split_vals:
        split_text = "--"
    elif np.isclose(min(split_vals), max(split_vals)):
        split_text = f"{min(split_vals):.4g}"
    else:
        split_text = f"{min(split_vals):.4g}-{max(split_vals):.4g}"

    return size_text, split_text


def print_hyperparameter_ranking(
    rows,
    metric="composite_score",
    n_best=20,
    success_quantile=0.10,
):
    """Rank (r_max, energy_weight, forces_weight) combinations by median metric."""
    required = ["r_max", "energy_weight", "forces_weight", metric]
    clean_rows = [r for r in rows if all(k in r for k in required) and np.isfinite(safe_float(r.get(metric)))]

    if not clean_rows:
        print()
        print(f"No rows available for hyperparameter ranking with metric: {metric}")
        return

    groups = defaultdict(list)
    for r in clean_rows:
        key = (safe_float(r["r_max"]), safe_float(r["energy_weight"]), safe_float(r["forces_weight"]))
        if all(np.isfinite(v) for v in key):
            groups[key].append(r)

    success_threshold = make_success_threshold(
        clean_rows,
        metric=metric,
        success_quantile=success_quantile,
    )

    records = []
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

        structures = sorted(set(str(r.get("structure", "--")) for r in group_rows))
        path_splits = sorted(set(str(r.get("path_split", r.get("split", "--"))) for r in group_rows))
        seeds = sorted(set(safe_int(r.get("seed")) for r in group_rows if safe_int(r.get("seed")) is not None))
        size_text, split_text = _size_split_text(group_rows)

        n_success = sum(safe_float(r.get(metric)) <= success_threshold for r in group_rows)

        records.append({
            "r_max": r_max,
            "energy_weight": ew,
            "forces_weight": fw,
            "n_runs": len(group_rows),
            "n_structures": len(structures),
            "n_path_splits": len(path_splits),
            "n_seeds": len(seeds),
            "size_text": size_text,
            "split_text": split_text,
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
        })

    records = sorted(records, key=lambda r: (r["median_score"], r["mean_score"], r["std_score"] if np.isfinite(r["std_score"]) else np.inf))

    print()
    print("=" * 150)
    print(f"HYPERPARAMETER RANKING BY MEDIAN {metric}")
    print("=" * 150)
    print(f"success quantile  : {success_quantile:.3f}")
    print(f"success threshold : {success_threshold:.6g}")
    print()

    header = (
        f"{'#':>3} {'r_max':>7} {'ew':>7} {'fw':>7} {'n':>5} {'struc':>6} {'paths':>7} {'seeds':>6} "
        f"{'size':>15} {'split_val':>15} {'median':>10} {'mean':>10} {'std':>10} {'best':>10} "
        f"{'succ.':>8} {'freqMAE':>10} {'specL2':>10} {'intR':>8} {'overlap':>9}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(records[:n_best], start=1):
        print(
            f"{i:>3d} {r['r_max']:>7.3g} {r['energy_weight']:>7.3g} {r['forces_weight']:>7.3g} "
            f"{r['n_runs']:>5d} {r['n_structures']:>6d} {r['n_path_splits']:>7d} {r['n_seeds']:>6d} "
            f"{r['size_text']:>15} {r['split_text']:>15} {r['median_score']:>10.4g} {r['mean_score']:>10.4g} "
            f"{r['std_score']:>10.4g} {r['best_score']:>10.4g} {100.0 * r['success_rate']:>7.1f}% "
            f"{r['median_freq_mae']:>10.4g} {r['median_spectrum_l2']:>10.4g} {r['median_intensity_r']:>8.3g} "
            f"{r['median_overlap']:>9.3g}"
        )

    print()
    print("LATEX TABLE")
    print("-" * 150)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{rrrrrrrrrrrr}")
    print(r"\hline")
    print(r"$r_\mathrm{max}$ & $w_E$ & $w_F$ & $n$ & $N$ & $s_\mathrm{val}$ & median & mean & std & best & freq. MAE & $r_I$ \\")
    print(r"\hline")
    for r in records[:n_best]:
        print(
            f"{format_latex_number(r['r_max'], 3)} & "
            f"{format_latex_number(r['energy_weight'], 3)} & "
            f"{format_latex_number(r['forces_weight'], 3)} & "
            f"{r['n_runs']} & "
            f"{r['size_text']} & "
            f"{r['split_text']} & "
            f"{format_latex_number(r['median_score'], 4)} & "
            f"{format_latex_number(r['mean_score'], 4)} & "
            f"{format_latex_number(r['std_score'], 3)} & "
            f"{format_latex_number(r['best_score'], 4)} & "
            f"{format_latex_number(r['median_freq_mae'], 4)} & "
            f"{format_latex_number(r['median_intensity_r'], 3)} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Hyperparameter ranking grouped by cutoff radius, energy weight, and force weight.}")
    print(r"\label{tab:hyperparameter_ranking}")
    print(r"\end{table}")


def print_pooled_hyperparameter_ranking(
    rows,
    metric="composite_score",
    success_quantile=0.10,
):
    """
    Main-effect / pooled parameter ranking.

    Authoritative pooled parameters:
        r_max, energy_weight, forces_weight, size, split_val
    """
    parameters = [
        ("r_max", r"$r_\mathrm{max}$"),
        ("energy_weight", r"$w_E$"),
        ("forces_weight", r"$w_F$"),
        ("size", r"$N$"),
        ("split_val", r"$s_\mathrm{val}$"),
    ]

    clean_rows = [r for r in rows if np.isfinite(safe_float(r.get(metric)))]
    if not clean_rows:
        print()
        print(f"No rows available for pooled parameter ranking with metric: {metric}")
        return

    success_threshold = make_success_threshold(
        clean_rows,
        metric=metric,
        success_quantile=success_quantile,
    )

    print()
    print("=" * 150)
    print(f"POOLED PARAMETER EFFECTS BY {metric}")
    print("=" * 150)
    print("lower is better    : yes")
    print(f"success quantile   : {success_quantile:.3f}")
    print(f"success threshold  : {success_threshold:.6g}")

    pooled_best = {}

    for param, latex_name in parameters:
        groups = defaultdict(list)
        for r in clean_rows:
            value = safe_float(r.get(param))
            if np.isfinite(value):
                groups[value].append(r)

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

            structures = sorted(set(str(r.get("structure", "--")) for r in group_rows))
            path_splits = sorted(set(str(r.get("path_split", r.get("split", "--"))) for r in group_rows))
            seeds = sorted(set(safe_int(r.get("seed")) for r in group_rows if safe_int(r.get("seed")) is not None))
            hparam_tuples = sorted(set(
                (
                    safe_float(r.get("size")),
                    safe_float(r.get("split_val")),
                    safe_float(r.get("r_max")),
                    safe_float(r.get("energy_weight")),
                    safe_float(r.get("forces_weight")),
                )
                for r in group_rows
                if all(np.isfinite(safe_float(r.get(k))) for k in ["size", "split_val", "r_max", "energy_weight", "forces_weight"])
            ))
            size_text, split_text = _size_split_text(group_rows)
            n_success = sum(safe_float(r.get(metric)) <= success_threshold for r in group_rows)

            records.append({
                "param": param,
                "value": value,
                "n_runs": len(group_rows),
                "n_structures": len(structures),
                "n_path_splits": len(path_splits),
                "n_seeds": len(seeds),
                "n_hparam_tuples": len(hparam_tuples),
                "size_text": size_text,
                "split_text": split_text,
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
            })

        records = sorted(records, key=lambda r: (r["median_score"], r["mean_score"], r["std_score"] if np.isfinite(r["std_score"]) else np.inf))
        pooled_best[param] = records[0]

        print()
        print("-" * 150)
        print(f"POOLED PARAMETER: {param}")
        print("-" * 150)

        header = (
            f"{'value':>10} {'n':>5} {'struc':>6} {'paths':>7} {'seeds':>6} {'tuples':>7} "
            f"{'size':>15} {'split_val':>15} {'median':>10} {'mean':>10} {'std':>10} {'best':>10} "
            f"{'worst':>10} {'succ.':>8} {'freqMAE':>10} {'specL2':>10} {'intR':>8} {'overlap':>9}"
        )
        print(header)
        print("-" * len(header))

        for r in records:
            print(
                f"{r['value']:>10.4g} {r['n_runs']:>5d} {r['n_structures']:>6d} {r['n_path_splits']:>7d} "
                f"{r['n_seeds']:>6d} {r['n_hparam_tuples']:>7d} {r['size_text']:>15} {r['split_text']:>15} "
                f"{r['median_score']:>10.4g} {r['mean_score']:>10.4g} {r['std_score']:>10.4g} "
                f"{r['best_score']:>10.4g} {r['worst_score']:>10.4g} {100.0 * r['success_rate']:>7.1f}% "
                f"{r['median_freq_mae']:>10.4g} {r['median_spectrum_l2']:>10.4g} {r['median_intensity_r']:>8.3g} "
                f"{r['median_overlap']:>9.3g}"
            )

    print()
    print("=" * 150)
    print("BEST POOLED VALUES")
    print("=" * 150)
    for param in ["r_max", "energy_weight", "forces_weight", "size", "split_val"]:
        if param not in pooled_best:
            continue
        r = pooled_best[param]
        print(f"{param:<15} = {r['value']:<8.4g} median={r['median_score']:.4g}, mean={r['mean_score']:.4g}, std={r['std_score']:.4g}, n={r['n_runs']}")


def print_probabilistic_hyperparameter_analysis(rows, *, metric="composite_score", success_quantile=0.10, ci=0.90, n_samples=20000):
    clean_rows = [r for r in rows if np.isfinite(safe_float(r.get(metric)))]
    if not clean_rows:
        print()
        print(f"No rows available for probabilistic analysis with metric: {metric}")
        return

    threshold = make_success_threshold(clean_rows, metric=metric, success_quantile=success_quantile)

    print()
    print("=" * 130)
    print("PROBABILISTIC HYPERPARAMETER SUCCESS ANALYSIS")
    print("=" * 130)
    print(f"metric             : {metric}")
    print("lower is better    : yes")
    print(f"success quantile   : {success_quantile:.3f}")
    print(f"success threshold  : {threshold:.6g}")
    print(f"credible interval  : {100 * ci:.1f}%")

    pooled_params = [
        ("r_max", "r_max"),
        ("energy_weight", "ew"),
        ("forces_weight", "fw"),
        ("size", "size"),
        ("split_val", "split"),
    ]

    for param, short_name in pooled_params:
        groups = defaultdict(list)
        for r in clean_rows:
            value = safe_float(r.get(param))
            if np.isfinite(value):
                groups[value].append(r)
        if not groups:
            continue

        records = []
        for value, group_rows in groups.items():
            summary = summarize_success_group(group_rows, metric=metric, threshold=threshold, ci=ci)
            records.append({"label": value, "param": param, "value": value, **summary})

        p_best = probability_best_from_beta(records, n_samples=n_samples, seed=123)
        for r in records:
            r["p_best"] = p_best.get(r["label"], np.nan)

        records = sorted(records, key=lambda r: (-r["posterior_mean"], -r["p_best"], r["median_score"]))

        print()
        print("-" * 130)
        print(f"POOLED PARAMETER: {param}")
        print("-" * 130)
        header = f"{short_name:>10} {'n':>5} {'succ':>6} {'raw':>8} {'post_mean':>10} {'CI_low':>10} {'CI_high':>10} {'P(best)':>10} {'median':>10} {'best':>10}"
        print(header)
        print("-" * len(header))
        for r in records:
            print(
                f"{r['value']:>10.4g} {r['n']:>5d} {r['successes']:>6d} {100 * r['raw_rate']:>7.1f}% "
                f"{100 * r['posterior_mean']:>9.1f}% {100 * r['ci_low']:>9.1f}% {100 * r['ci_high']:>9.1f}% "
                f"{100 * r['p_best']:>9.1f}% {r['median_score']:>10.4g} {r['best_score']:>10.4g}"
            )

    tuple_groups = defaultdict(list)
    for r in clean_rows:
        key = (
            safe_float(r.get("size")),
            safe_float(r.get("split_val")),
            safe_float(r.get("r_max")),
            safe_float(r.get("energy_weight")),
            safe_float(r.get("forces_weight")),
        )
        if all(np.isfinite(v) for v in key):
            tuple_groups[key].append(r)

    tuple_records = []
    for key, group_rows in tuple_groups.items():
        size, split_val, r_max, ew, fw = key
        summary = summarize_success_group(group_rows, metric=metric, threshold=threshold, ci=ci)
        tuple_records.append({
            "label": key,
            "size": size,
            "split_val": split_val,
            "r_max": r_max,
            "energy_weight": ew,
            "forces_weight": fw,
            **summary,
        })

    p_best_tuple = probability_best_from_beta(tuple_records, n_samples=n_samples, seed=456)
    for r in tuple_records:
        r["p_best"] = p_best_tuple.get(r["label"], np.nan)
    tuple_records = sorted(tuple_records, key=lambda r: (-r["posterior_mean"], -r["p_best"], r["median_score"]))

    print()
    print("-" * 130)
    print("FULL TUPLE PROBABILITIES")
    print("-" * 130)
    header = f"{'size':>8} {'split':>8} {'r_max':>7} {'ew':>7} {'fw':>7} {'n':>5} {'succ':>6} {'raw':>8} {'post_mean':>10} {'CI_low':>10} {'CI_high':>10} {'P(best)':>10} {'median':>10} {'best':>10}"
    print(header)
    print("-" * len(header))
    for r in tuple_records[:20]:
        print(
            f"{r['size']:>8.4g} {r['split_val']:>8.4g} {r['r_max']:>7.3g} {r['energy_weight']:>7.3g} {r['forces_weight']:>7.3g} "
            f"{r['n']:>5d} {r['successes']:>6d} {100 * r['raw_rate']:>7.1f}% {100 * r['posterior_mean']:>9.1f}% "
            f"{100 * r['ci_low']:>9.1f}% {100 * r['ci_high']:>9.1f}% {100 * r['p_best']:>9.1f}% "
            f"{r['median_score']:>10.4g} {r['best_score']:>10.4g}"
        )


def print_parameter_source_summary(rows, parameters=None):
    if parameters is None:
        parameters = ["r_max", "energy_weight", "forces_weight", "seed", "size", "split_val"]

    print()
    print("PARAMETER SOURCE / CONSISTENCY SUMMARY")
    print("-" * 120)
    for param in parameters:
        source_counts = defaultdict(int)
        status_counts = defaultdict(int)
        for r in rows:
            source_counts[str(r.get(f"{param}_source", "missing"))] += 1
            status_counts[str(r.get(f"{param}_status", "missing"))] += 1
        print()
        print(f"{param}")
        print("  sources :", dict(sorted(source_counts.items())))
        print("  status  :", dict(sorted(status_counts.items())))

    mismatch_rows = []
    for r in rows:
        for param in parameters:
            if r.get(f"{param}_status") == "mismatch":
                mismatch_rows.append((param, r))

    if mismatch_rows:
        print()
        print("PARAMETER MISMATCHES")
        print("-" * 120)
        header = f"{'param':<15} {'structure':<15} {'path_split':<30} {'db':>12} {'run_id':>12} {'path':>12} {'run_id_full'}"
        print(header)
        print("-" * len(header))
        for param, r in mismatch_rows[:40]:
            path_value = r.get("path_size_split", "--") if param == "size" else "--"
            print(
                f"{param:<15} {str(r.get('structure', '--')):<15} {str(r.get('path_split', '--')):<30.30} "
                f"{str(r.get('db_' + param, '--')):>12} {str(r.get('run_id_' + param, '--')):>12} {str(path_value):>12} "
                f"{str(r.get('run_id', '--'))}"
            )
        if len(mismatch_rows) > 40:
            print(f"... {len(mismatch_rows) - 40} more mismatches not shown")


def print_size_interaction_ranking(rows, metric="composite_score", n_best=30):
    clean_rows = [r for r in rows if np.isfinite(safe_float(r.get(metric)))]
    if not clean_rows:
        print()
        print(f"No rows available for size interaction ranking with metric: {metric}")
        return

    interaction_specs = [
        ("SIZE x R_MAX", ["size", "r_max"]),
        ("SPLIT_VAL x R_MAX", ["split_val", "r_max"]),
        ("SIZE x SPLIT_VAL", ["size", "split_val"]),
        ("SIZE x SPLIT_VAL x FULL HYPERPARAMETER TUPLE", ["size", "split_val", "r_max", "energy_weight", "forces_weight"]),
    ]

    print()
    print("=" * 150)
    print(f"SIZE / SPLIT_VAL INTERACTION RANKING BY {metric}")
    print("=" * 150)

    for title, keys in interaction_specs:
        groups = defaultdict(list)
        for r in clean_rows:
            values = tuple(safe_float(r.get(key)) for key in keys)
            if all(np.isfinite(v) for v in values):
                groups[values].append(r)
        if not groups:
            continue

        records = []
        for key_tuple, group_rows in groups.items():
            scores = [safe_float(r.get(metric)) for r in group_rows]
            structures = sorted(set(str(r.get("structure", "--")) for r in group_rows))
            path_splits = sorted(set(str(r.get("path_split", r.get("split", "--"))) for r in group_rows))
            seeds = sorted(set(safe_int(r.get("seed")) for r in group_rows if safe_int(r.get("seed")) is not None))
            records.append({
                "key_tuple": key_tuple,
                "n_runs": len(group_rows),
                "n_structures": len(structures),
                "n_path_splits": len(path_splits),
                "n_seeds": len(seeds),
                "median_score": median_or_nan(scores),
                "mean_score": mean_or_nan(scores),
                "std_score": std_or_nan(scores),
                "best_score": min_or_nan(scores),
                "worst_score": max_or_nan(scores),
            })

        records = sorted(records, key=lambda r: (r["median_score"], r["mean_score"], r["std_score"] if np.isfinite(r["std_score"]) else np.inf))

        print()
        print("-" * 150)
        print(title)
        print("-" * 150)
        key_header = " ".join(f"{k:>12}" for k in keys)
        header = f"{'#':>3} {key_header} {'n':>5} {'struc':>6} {'paths':>7} {'seeds':>6} {'median':>10} {'mean':>10} {'std':>10} {'best':>10} {'worst':>10}"
        print(header)
        print("-" * len(header))
        for i, r in enumerate(records[:n_best], start=1):
            key_text = " ".join(f"{v:>12.4g}" for v in r["key_tuple"])
            print(
                f"{i:>3d} {key_text} {r['n_runs']:>5d} {r['n_structures']:>6d} {r['n_path_splits']:>7d} "
                f"{r['n_seeds']:>6d} {r['median_score']:>10.4g} {r['mean_score']:>10.4g} {r['std_score']:>10.4g} "
                f"{r['best_score']:>10.4g} {r['worst_score']:>10.4g}"
            )


# ============================================================
# Row collection and summaries
# ============================================================

def collect_model_rows_for_structure(h5, structure: str, *, include_all_splits: bool = False):
    all_runs = list(iter_evaluation_runs(h5, structure) or [])
    if include_all_splits:
        runs = all_runs
    else:
        runs = [row for row in all_runs if split_matches_structure(row[0], structure)]

    model_rows = []
    for path_split, sweep_id, run_id, run_group in runs:
        model_rows.append(make_analysis_row(
            structure=structure,
            path_split=path_split,
            sweep_id=sweep_id,
            run_id=run_id,
            run_group=run_group,
        ))

    return model_rows, len(all_runs), len(runs)


def summarize_structure(
    h5,
    structure: str,
    include_all_splits: bool = False,
    rank_hyperparams: bool = False,
    pool_hyperparams: bool = False,
    prob_hyperparams: bool = False,
    size_interactions: bool = False,
    rank_metric: str = "composite_score",
    rank_top: int = 20,
    success_quantile: float = 0.10,
    prob_ci: float = 0.90,
    prob_samples: int = 20000,
):
    all_runs = list(iter_evaluation_runs(h5, structure) or [])
    runs = all_runs if include_all_splits else [row for row in all_runs if split_matches_structure(row[0], structure)]

    if not runs:
        print(f"No evaluations found for structure: {structure}")
        return

    param_values = defaultdict(set)
    metric_values = defaultdict(list)
    path_split_values = set()
    sweep_values = set()
    missing_param_counts = defaultdict(int)
    model_rows = []

    expected_params = ["model_type", "batch_size", "max_epochs", "energy_weight", "forces_weight", "r_max", "seed", "size", "split_val"]

    for path_split, sweep_id, run_id, run_group in runs:
        path_split_values.add(path_split)
        sweep_values.add((path_split, sweep_id))

        row = make_analysis_row(
            structure=structure,
            path_split=path_split,
            sweep_id=sweep_id,
            run_id=run_id,
            run_group=run_group,
        )
        model_rows.append(row)

        for key in expected_params:
            value = row.get(key)
            if value is None:
                missing_param_counts[key] += 1
            else:
                param_values[key].add(value)

        for key, value in collect_metric_attrs(run_group).items():
            add_if_numeric(metric_values, key, value)

        if "imag_flags" in run_group:
            metric_values["n_imag_model"].append(float(int(np.sum(run_group["imag_flags"][()]))))

    print("=" * 110)
    print(f"REF DB SWEEP EXPLORER: {structure}")
    print("=" * 110)
    print(f"model evaluations : {len(runs)}")
    print(f"filtered out       : {len(all_runs) - len(runs)}")
    print(f"path splits        : {len(path_split_values)}")
    print(f"sweeps             : {len(sweep_values)}")
    print()

    print("PATH SPLITS (TRACEABILITY ONLY)")
    print("-" * 110)
    for path_split in sorted(path_split_values):
        n = sum(1 for s, _, _, _ in runs if s == path_split)
        print(f"{path_split:<35} {n:>6} runs")

    print()
    print("ANALYSIS PARAMETERS AFTER RESOLUTION")
    print("-" * 110)
    header = f"{'parameter':<20} {'n_unique':>10}  values"
    print(header)
    print("-" * len(header))
    for key in sorted(param_values):
        values = sorted(param_values[key], key=lambda x: str(x))
        print(f"{key:<20} {len(values):>10}  {values}")

    print()
    print("MISSING PARAMETER COUNTS")
    print("-" * 110)
    any_missing = False
    for key in sorted(missing_param_counts):
        count = missing_param_counts[key]
        if count > 0:
            any_missing = True
            print(f"{key:<20} {count:>10} / {len(runs)}")
    if not any_missing:
        print("none")

    print()
    print("METRICS FOUND IN ranking_metrics")
    print("-" * 110)
    header = f"{'metric':<35} {'n':>6} {'min':>14} {'max':>14} {'mean':>14}"
    print(header)
    print("-" * len(header))
    for key in sorted(metric_values):
        values = np.asarray(metric_values[key], dtype=float)
        print(f"{key:<35} {len(values):>6d} {np.nanmin(values):>14.6g} {np.nanmax(values):>14.6g} {np.nanmean(values):>14.6g}")

    print_parameter_source_summary(model_rows)
    print_best_models(model_rows, n=7, metric="composite_score")

    if rank_hyperparams:
        print_hyperparameter_ranking(model_rows, metric=rank_metric, n_best=rank_top, success_quantile=success_quantile)
    if pool_hyperparams:
        print_pooled_hyperparameter_ranking(model_rows, metric=rank_metric, success_quantile=success_quantile)
    if size_interactions:
        print_size_interaction_ranking(model_rows, metric=rank_metric, n_best=rank_top)
    if prob_hyperparams:
        print_probabilistic_hyperparameter_analysis(model_rows, metric=rank_metric, success_quantile=success_quantile, ci=prob_ci, n_samples=prob_samples)


def summarize_combined_structures(
    h5,
    structures,
    *,
    include_all_splits=False,
    rank_hyperparams=False,
    pool_hyperparams=False,
    prob_hyperparams=False,
    size_interactions=False,
    rank_metric="composite_score",
    normalize_score="percentile",
    rank_top=20,
    success_quantile=0.10,
    prob_ci=0.90,
    prob_samples=20000,
):
    all_rows = []
    total_all_runs = 0
    total_kept_runs = 0

    for structure in structures:
        rows, n_all, n_kept = collect_model_rows_for_structure(h5, structure, include_all_splits=include_all_splits)
        all_rows.extend(rows)
        total_all_runs += n_all
        total_kept_runs += n_kept

    if not all_rows:
        print(f"No evaluations found for combined structures: {structures}")
        return

    if len(structures) > 1 and normalize_score == "none":
        print()
        print("WARNING: pooling several structures with raw scores.")
        print("         This is usually not recommended unless scores are directly comparable.")

    if normalize_score != "none":
        analysis_metric = add_normalized_metric(
            all_rows,
            metric=rank_metric,
            group_key="structure",
            method=normalize_score,
            out_key=f"{rank_metric}_{normalize_score}",
        )
    else:
        analysis_metric = rank_metric

    structure_counts = defaultdict(int)
    for r in all_rows:
        structure_counts[r.get("structure", "--")] += 1

    print("=" * 130)
    print("COMBINED REF DB SWEEP EXPLORER")
    print("=" * 130)
    print(f"structures          : {', '.join(structures)}")
    print(f"raw metric          : {rank_metric}")
    print(f"analysis metric     : {analysis_metric}")
    print(f"normalization       : {normalize_score}")
    print(f"model evaluations   : {total_kept_runs}")
    print(f"filtered out        : {total_all_runs - total_kept_runs}")
    print()

    print("STRUCTURE COUNTS")
    print("-" * 130)
    for structure in structures:
        print(f"{structure:<30} {structure_counts[structure]:>6} runs")

    print()
    print("METRIC INTERPRETATION")
    print("-" * 130)
    if normalize_score == "percentile":
        print("0.0 = best run within its structure, 1.0 = worst run within its structure.")
    elif normalize_score == "zscore":
        print("Negative values are better than the structure mean; positive values are worse.")
    elif normalize_score == "relative_best":
        print("1.0 = best run within its structure; larger values are worse.")
    else:
        print("Raw metric values are used directly.")

    print_parameter_source_summary(all_rows)
    print_best_models(all_rows, n=7, metric=analysis_metric)

    if rank_hyperparams:
        print_hyperparameter_ranking(all_rows, metric=analysis_metric, n_best=rank_top, success_quantile=success_quantile)
    if pool_hyperparams:
        print_pooled_hyperparameter_ranking(all_rows, metric=analysis_metric, success_quantile=success_quantile)
    if size_interactions:
        print_size_interaction_ranking(all_rows, metric=analysis_metric, n_best=rank_top)
    if prob_hyperparams:
        print_probabilistic_hyperparameter_analysis(all_rows, metric=analysis_metric, success_quantile=success_quantile, ci=prob_ci, n_samples=prob_samples)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore available sweep parameters and metrics in ref_db.h5.")
    parser.add_argument("--ref_db", type=Path, default=REF_PATH)

    structure_group = parser.add_mutually_exclusive_group(required=True)
    structure_group.add_argument("--structure", help="Single structure to analyse, e.g. SiO2 or SiO2_PBE.")
    structure_group.add_argument("--structures", nargs="+", help="Several structures to analyse together. Use '--structures all' for all structures.")

    parser.add_argument(
        "--include_all_splits",
        action="store_true",
        help="Include all path-level splits under the structure group. Path split is traceability only.",
    )
    parser.add_argument("--rank-hyperparams", action="store_true", help="Rank r_max, energy_weight, forces_weight combinations by median score.")
    parser.add_argument("--rank-metric", default="composite_score", help="Metric used for hyperparameter ranking.")
    parser.add_argument("--rank-top", type=int, default=20, help="Number of combinations to print.")
    parser.add_argument("--pool-hyperparams", action="store_true", help="Print pooled main-effect ranking for r_max, energy_weight, forces_weight, size, and split_val.")
    parser.add_argument("--size-interactions", action="store_true", help="Print size/split_val interaction rankings.")
    parser.add_argument("--prob_hyperparams", action="store_true", help="Estimate probability that parameters yield top-performing models.")
    parser.add_argument("--success_quantile", type=float, default=0.10, help="Success threshold as lower quantile of the ranking metric.")
    parser.add_argument("--prob_ci", type=float, default=0.90, help="Credible interval width for Beta-binomial probabilities.")
    parser.add_argument("--prob_samples", type=int, default=20000, help="Monte Carlo samples for estimating P(best).")
    parser.add_argument(
        "--normalize-score",
        choices=["none", "percentile", "zscore", "relative_best"],
        default="percentile",
        help="Score normalization for combined multi-structure analysis. Ignored for single-structure analysis.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with h5py.File(args.ref_db, "r") as h5:
        if "structures" not in h5:
            raise KeyError("Missing top-level group: /structures")

        available_structures = sorted(h5["structures"].keys())

        if args.structure is not None:
            if args.structure not in h5["structures"]:
                raise KeyError(f"Structure not found: {args.structure}")

            summarize_structure(
                h5,
                args.structure,
                include_all_splits=args.include_all_splits,
                rank_hyperparams=args.rank_hyperparams,
                pool_hyperparams=args.pool_hyperparams,
                prob_hyperparams=args.prob_hyperparams,
                size_interactions=args.size_interactions,
                rank_metric=args.rank_metric,
                rank_top=args.rank_top,
                success_quantile=args.success_quantile,
                prob_ci=args.prob_ci,
                prob_samples=args.prob_samples,
            )

        else:
            if len(args.structures) == 1 and args.structures[0].lower() == "all":
                structures = available_structures
            else:
                structures = args.structures

            missing = [s for s in structures if s not in h5["structures"]]
            if missing:
                raise KeyError(
                    "Structures not found: "
                    + ", ".join(missing)
                    + "\nAvailable structures: "
                    + ", ".join(available_structures)
                )

            summarize_combined_structures(
                h5,
                structures,
                include_all_splits=args.include_all_splits,
                rank_hyperparams=args.rank_hyperparams,
                pool_hyperparams=args.pool_hyperparams,
                prob_hyperparams=args.prob_hyperparams,
                size_interactions=args.size_interactions,
                rank_metric=args.rank_metric,
                normalize_score=args.normalize_score,
                rank_top=args.rank_top,
                success_quantile=args.success_quantile,
                prob_ci=args.prob_ci,
                prob_samples=args.prob_samples,
            )


if __name__ == "__main__":
    main()
