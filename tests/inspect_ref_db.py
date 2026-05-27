#!/usr/bin/env python3
from pathlib import Path
import argparse
import h5py
import numpy as np

REF_PATH = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5")


def print_attrs(obj, indent=""):
    if not obj.attrs:
        return
    print(f"{indent}attrs:")
    for k, v in obj.attrs.items():
        print(f"{indent}  {k}: {v}")


def describe_dataset(name, ds, indent=""):
    data = ds[()]
    print(f"{indent}{name}: shape={ds.shape}, dtype={ds.dtype}")

    if np.issubdtype(ds.dtype, np.number) and data.size > 0:
        arr = np.asarray(data)
        print(
            f"{indent}  min={np.nanmin(arr):.6g}, "
            f"max={np.nanmax(arr):.6g}, "
            f"mean={np.nanmean(arr):.6g}"
        )


def recurse_group(group, indent=""):
    print_attrs(group, indent=indent)

    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            describe_dataset(name, obj, indent=indent)
        elif isinstance(obj, h5py.Group):
            print(f"{indent}{name}/")
            recurse_group(obj, indent=indent + "  ")


def get_ref_basic_stats(sg):
    out = {
        "has_crystal": "crystal" in sg,
        "natoms": None,
        "n_modes": None,
        "n_imag": None,
        "has_bec": False,
        "has_dielectric": False,
    }

    if "crystal" not in sg:
        return out

    cg = sg["crystal"]

    if "geometry/atomic_numbers" in cg:
        out["natoms"] = len(cg["geometry/atomic_numbers"])

    if "frequencies_cm1" in cg:
        out["n_modes"] = len(cg["frequencies_cm1"])

    if "imag_flags" in cg:
        out["n_imag"] = int(np.sum(cg["imag_flags"][()]))

    out["has_bec"] = "born_charges" in cg
    out["has_dielectric"] = "dielectric_tensor" in cg

    return out


def iter_evaluation_runs(sg):
    if "evaluations" not in sg:
        return

    eg_root = sg["evaluations"]

    for dataset_split, split_group in eg_root.items():
        if not isinstance(split_group, h5py.Group):
            continue

        for sweep_id, sweep_group in split_group.items():
            if not isinstance(sweep_group, h5py.Group):
                continue

            for run_id, run_group in sweep_group.items():
                if not isinstance(run_group, h5py.Group):
                    continue

                yield dataset_split, sweep_id, run_id, run_group


def get_metric(run_group, key, default=None):
    if "ranking_metrics" not in run_group:
        return default

    rg = run_group["ranking_metrics"]

    if key in rg.attrs:
        value = rg.attrs[key]
        try:
            return float(value)
        except Exception:
            return value

    return default


def summarize_evaluation_counts(sg):
    splits = set()
    sweeps = set()
    runs = []

    for dataset_split, sweep_id, run_id, run_group in iter_evaluation_runs(sg) or []:
        splits.add(dataset_split)
        sweeps.add((dataset_split, sweep_id))
        runs.append((dataset_split, sweep_id, run_id, run_group))

    return {
        "n_splits": len(splits),
        "n_sweeps": len(sweeps),
        "n_runs": len(runs),
        "runs": runs,
    }


def find_best_run(runs, metric="composite_score"):
    best = None

    for dataset_split, sweep_id, run_id, run_group in runs:
        score = get_metric(run_group, metric)

        if score is None:
            continue

        record = {
            "dataset_split": dataset_split,
            "sweep_id": sweep_id,
            "run_id": run_id,
            "score": score,
            "run_group": run_group,
        }

        if best is None or score < best["score"]:
            best = record

    return best


def print_structure_summary(h5, structure):
    path = f"structures/{structure}"

    if path not in h5:
        print(f"{structure}: MISSING")
        return False

    sg = h5[path]
    ref = get_ref_basic_stats(sg)
    ev = summarize_evaluation_counts(sg)

    print("=" * 80)
    print(f"STRUCTURE: {structure}")
    print("=" * 80)

    print(f"  crystal reference : {'OK' if ref['has_crystal'] else 'MISSING'}")
    print(f"  natoms            : {ref['natoms']}")
    print(f"  n_modes           : {ref['n_modes']}")
    print(f"  n_imag_ref        : {ref['n_imag']}")
    print(f"  BEC               : {'OK' if ref['has_bec'] else 'missing'}")
    print(f"  dielectric        : {'OK' if ref['has_dielectric'] else 'missing'}")
    print(f"  evaluation splits : {ev['n_splits']}")
    print(f"  sweeps            : {ev['n_sweeps']}")
    print(f"  model evaluations : {ev['n_runs']}")

    best = find_best_run(ev["runs"])

    if best is not None:
        print()
        print("  best by composite_score:")
        print(f"    split : {best['dataset_split']}")
        print(f"    sweep : {best['sweep_id']}")
        print(f"    run   : {best['run_id']}")
        print(f"    score : {best['score']:.6g}")

    return ref["has_crystal"]


def print_evaluations_table(h5, structure):
    path = f"structures/{structure}"

    if path not in h5:
        raise KeyError(f"Missing structure: {structure}")

    sg = h5[path]
    runs = list(iter_evaluation_runs(sg) or [])

    print("=" * 120)
    print(f"EVALUATIONS: {structure}")
    print("=" * 120)

    if not runs:
        print("No model evaluations stored.")
        return

    header = (
        f"{'dataset_split':<18} "
        f"{'sweep_id':<34} "
        f"{'run_id':<45} "
        f"{'score':>10} "
        f"{'freq_mae':>10} "
        f"{'spec_l2':>10} "
        f"{'n_imag':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []

    for dataset_split, sweep_id, run_id, run_group in runs:
        score = get_metric(run_group, "composite_score")
        freq_mae = get_metric(run_group, "freq_mae_ir_cm1")
        spec_l2 = get_metric(run_group, "spectrum_rel_l2")

        n_imag = None
        if "imag_flags" in run_group:
            n_imag = int(np.sum(run_group["imag_flags"][()]))

        rows.append((score if score is not None else np.inf, dataset_split, sweep_id, run_id, score, freq_mae, spec_l2, n_imag))

    rows.sort(key=lambda x: x[0])

    for _, dataset_split, sweep_id, run_id, score, freq_mae, spec_l2, n_imag in rows:
        print(
            f"{dataset_split:<18.18} "
            f"{sweep_id:<34.34} "
            f"{run_id:<45.45} "
            f"{score if score is not None else np.nan:10.4g} "
            f"{freq_mae if freq_mae is not None else np.nan:10.4g} "
            f"{spec_l2 if spec_l2 is not None else np.nan:10.4g} "
            f"{n_imag if n_imag is not None else -1:7d}"
        )


def print_details(h5, structure):
    path = f"structures/{structure}"

    if path not in h5:
        print(f"[MISSING] {path}")
        return False

    sg = h5[path]
    print("=" * 80)
    print(f"DETAILS: {structure}")
    print("=" * 80)

    recurse_group(sg, indent="  ")
    return True


def main():
    parser = argparse.ArgumentParser(description="Inspect and sanity-check ref_db.h5.")
    parser.add_argument("--ref_db", type=Path, default=REF_PATH, help="Path to ref_db.h5")
    parser.add_argument("--structure", default=None, help="Structure name to inspect")
    parser.add_argument("--list", action="store_true", help="Only list structures")
    parser.add_argument("--summary", action="store_true", help="Print compact structure summary")
    parser.add_argument("--details", action="store_true", help="Print full recursive DB details")
    parser.add_argument("--evaluations", action="store_true", help="Print compact evaluation table")
    args = parser.parse_args()

    with h5py.File(args.ref_db, "r") as h5:
        if "structures" not in h5:
            raise KeyError("Missing top-level group: /structures")

        structures = sorted(h5["structures"].keys())

        print(f"DB: {args.ref_db}")
        print(f"Structures: {len(structures)}")

        if args.list:
            for s in structures:
                print(f"  - {s}")
            return

        selected = [args.structure] if args.structure is not None else structures

        for structure in selected:
            if structure not in structures:
                raise KeyError(f"Structure not found: {structure}")

        if args.details:
            for structure in selected:
                print_details(h5, structure)
            return

        if args.evaluations:
            for structure in selected:
                print_evaluations_table(h5, structure)
            return

        # Default behavior: compact summary.
        for structure in selected:
            print_structure_summary(h5, structure)


if __name__ == "__main__":
    main()
