#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import re
from collections import defaultdict

import h5py
import numpy as np

REF_PATH = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5")


def parse_float_token(value: str) -> float:
    value = value.replace("p", ".")
    return float(value)


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
    }

    out = {}

    for key, pat in patterns.items():
        m = re.search(pat, run_id)
        if not m:
            continue

        value = m.group(1)

        if key in {"batch_size", "max_epochs", "seed"}:
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



def add_if_numeric(store, key, value):
    try:
        value = float(value)
    except Exception:
        return
    if np.isfinite(value):
        store[key].append(value)


def summarize_structure(h5, structure: str, include_all_splits: bool = False):
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
            **params,
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

def main():
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
        )


if __name__ == "__main__":
    main()

