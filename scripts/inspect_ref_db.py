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


def decode_h5_value(value):
    """
    Convert HDF5 byte strings to normal Python strings where useful.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)

    return value


def describe_dataset(
    name,
    ds,
    indent="",
    *,
    print_values=False,
    max_full_size=50,
    preview_size=10,
    precision=6,
):
    """
    Print dataset shape/dtype and optionally values.

    For small arrays:
        prints full value.

    For large numeric arrays:
        prints first N values and min/max/mean.

    For large non-numeric arrays:
        prints first N values only.
    """
    print(f"{indent}{name}: shape={ds.shape}, dtype={ds.dtype}")

    print_attrs(ds, indent=indent + "  ")

    if not print_values:
        return

    value = decode_h5_value(ds[()])

    if isinstance(value, np.ndarray):
        if value.size <= max_full_size:
            with np.printoptions(
                precision=precision,
                suppress=True,
                linewidth=160,
            ):
                print(f"{indent}  value = {value}")
            return

        flat = value.ravel()
        preview = flat[:preview_size]

        with np.printoptions(
            precision=precision,
            suppress=True,
            linewidth=160,
        ):
            print(f"{indent}  preview first {preview_size} = {preview}")

        if np.issubdtype(value.dtype, np.number):
            finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.floating) else value

            if finite.size > 0:
                print(
                    f"{indent}  min={np.nanmin(value):.6g}, "
                    f"max={np.nanmax(value):.6g}, "
                    f"mean={np.nanmean(value):.6g}"
                )
        return

    print(f"{indent}  value = {value}")


def recurse_group(
    group,
    indent="",
    *,
    print_values=False,
    max_full_size=50,
    preview_size=10,
    precision=6,
):
    print_attrs(group, indent=indent)

    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            describe_dataset(
                name,
                obj,
                indent=indent,
                print_values=print_values,
                max_full_size=max_full_size,
                preview_size=preview_size,
                precision=precision,
            )

        elif isinstance(obj, h5py.Group):
            print(f"{indent}{name}/")
            recurse_group(
                obj,
                indent=indent + "  ",
                print_values=print_values,
                max_full_size=max_full_size,
                preview_size=preview_size,
                precision=precision,
            )


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


def evaluation_group_path(structure, split, sweep_id, run_id):
    return f"structures/{structure}/evaluations/{split}/{sweep_id}/{run_id}"


def read_dataset(group, name):
    if name not in group:
        raise KeyError(f"Missing dataset: {group.name}/{name}")
    return np.asarray(group[name][()])


def print_array(name, arr):
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    print(arr)


def compare_numeric_arrays(ref, model, label="array"):
    ref = np.asarray(ref, dtype=float)
    model = np.asarray(model, dtype=float)

    print(f"{label}:")
    print(f"  ref shape   : {ref.shape}")
    print(f"  model shape : {model.shape}")

    if ref.shape != model.shape:
        print("  shape match : NO")
        return

    diff = model - ref
    print("  shape match : yes")
    print(f"  MAE         : {np.mean(np.abs(diff)):.6g}")
    print(f"  RMSE        : {np.sqrt(np.mean(diff**2)):.6g}")
    print(f"  max abs     : {np.max(np.abs(diff)):.6g}")
    print(f"  mean signed : {np.mean(diff):.6g}")


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


def print_run_layout(
    h5,
    *,
    structure,
    split,
    sweep_id,
    run_id,
    print_values=False,
    max_full_size=50,
    preview_size=10,
    precision=6,
):
    """
    Print the complete DB layout for exactly one model evaluation run.
    """
    path = evaluation_group_path(structure, split, sweep_id, run_id)

    print("=" * 100)
    print("RUN DB LAYOUT")
    print("=" * 100)
    print(f"DB group  : {path}")
    print(f"structure : {structure}")
    print(f"split     : {split}")
    print(f"sweep_id  : {sweep_id}")
    print(f"run_id    : {run_id}")
    print(f"values    : {print_values}")
    print("=" * 100)

    if path not in h5:
        raise KeyError(f"Missing model evaluation group: {path}")

    run_group = h5[path]

    recurse_group(
        run_group,
        indent="  ",
        print_values=print_values,
        max_full_size=max_full_size,
        preview_size=preview_size,
        precision=precision,
    )


def print_frequency_comparison(ref_freqs, model_freqs):
    ref_freqs = np.asarray(ref_freqs, dtype=float)
    model_freqs = np.asarray(model_freqs, dtype=float)

    n = min(len(ref_freqs), len(model_freqs))

    print("=" * 80)
    print("FREQUENCY COMPARISON")
    print("=" * 80)
    print(f"n_ref   : {len(ref_freqs)}")
    print(f"n_model : {len(model_freqs)}")
    print(f"n_used  : {n}")
    print()

    diff = model_freqs[:n] - ref_freqs[:n]

    print(f"MAE         : {np.mean(np.abs(diff)):.6g} cm^-1")
    print(f"RMSE        : {np.sqrt(np.mean(diff**2)):.6g} cm^-1")
    print(f"max abs     : {np.max(np.abs(diff)):.6g} cm^-1")
    print(f"mean signed : {np.mean(diff):.6g} cm^-1")
    print()

    header = (
        f"{'mode':>5} "
        f"{'ref_cm1':>14} "
        f"{'model_cm1':>14} "
        f"{'delta_cm1':>14} "
        f"{'abs_delta':>14}"
    )
    print(header)
    print("-" * len(header))

    for i in range(n):
        d = model_freqs[i] - ref_freqs[i]
        print(
            f"{i:5d} "
            f"{ref_freqs[i]:14.6f} "
            f"{model_freqs[i]:14.6f} "
            f"{d:14.6f} "
            f"{abs(d):14.6f}"
        )


def compare_property(
    h5,
    *,
    structure,
    split,
    sweep_id,
    run_id,
    prop,
    fmt="auto",
):
    crystal_path = f"structures/{structure}/crystal"
    eval_path = evaluation_group_path(structure, split, sweep_id, run_id)

    if crystal_path not in h5:
        raise KeyError(f"Missing CRYSTAL reference group: {crystal_path}")

    if eval_path not in h5:
        raise KeyError(f"Missing model evaluation group: {eval_path}")

    cg = h5[crystal_path]
    eg = h5[eval_path]

    prop = prop.lower()

    mappings = {
        "freqs": ("frequencies_cm1", "frequencies_cm1"),
        "frequencies": ("frequencies_cm1", "frequencies_cm1"),
        "bec": ("born_charges", "bec_asr"),
        "born": ("born_charges", "bec_asr"),
        "intensities": ("intensities_km_mol", "intensities"),
        "imag_flags": ("imag_flags", "imag_flags"),
        "eigvals": ("eigvals_SI", "eigvals_SI"),
        "eigvecs": ("eigvecs_mw", "eigvecs_mw"),
    }

    if prop not in mappings:
        raise ValueError(
            f"Unknown property: {prop}. "
            f"Available: {', '.join(sorted(mappings))}"
        )

    ref_name, model_name = mappings[prop]

    ref = read_dataset(cg, ref_name)
    model = read_dataset(eg, model_name)

    print("=" * 80)
    print("COMPARE")
    print("=" * 80)
    print(f"structure : {structure}")
    print(f"split     : {split}")
    print(f"sweep_id  : {sweep_id}")
    print(f"run_id    : {run_id}")
    print(f"property  : {prop}")
    print(f"reference : {crystal_path}/{ref_name}")
    print(f"model     : {eval_path}/{model_name}")
    print()

    if fmt == "auto":
        if prop in {"freqs", "frequencies"}:
            fmt = "table"
        elif prop in {"bec", "born", "eigvecs"}:
            fmt = "summary"
        else:
            fmt = "summary"

    if fmt == "arrays":
        print_array("reference", ref)
        print()
        print_array("model", model)
        return

    if prop in {"freqs", "frequencies"} and fmt == "table":
        print_frequency_comparison(ref, model)
        return

    compare_numeric_arrays(ref, model, label=prop)


def main():
    parser = argparse.ArgumentParser(description="Inspect and sanity-check ref_db.h5.")
    parser.add_argument("--ref_db", type=Path, default=REF_PATH, help="Path to ref_db.h5")
    parser.add_argument("--structure", default=None, help="Structure name to inspect")
    parser.add_argument("--list", action="store_true", help="Only list structures")
    parser.add_argument("--summary", action="store_true", help="Print compact structure summary")
    parser.add_argument("--details", action="store_true", help="Print full recursive DB details")
    parser.add_argument("--evaluations", action="store_true", help="Print compact evaluation table")
    parser.add_argument(
        "--run-layout",
        action="store_true",
        help="Print full recursive DB layout for one evaluation run.",
    )

    parser.add_argument(
        "--values",
        action="store_true",
        help="When printing layouts/details, also print dataset values or previews.",
    )

    parser.add_argument(
        "--max-full-size",
        type=int,
        default=50,
        help="Maximum array size printed fully when --values is active.",
    )

    parser.add_argument(
        "--preview-size",
        type=int,
        default=10,
        help="Number of flattened values previewed for large arrays when --values is active.",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Floating point precision for printed arrays.",
    )
    parser.add_argument("--compare", action="store_true", help="Compare model property against CRYSTAL reference")
    parser.add_argument("--split", default=None, help="Dataset split for comparison")
    parser.add_argument("--sweep_id", default=None, help="Sweep id for comparison")
    parser.add_argument("--run_id", default=None, help="Run id / model id for comparison")
    parser.add_argument(
        "--property",
        default="freqs",
        help="Property to compare: freqs, bec, intensities, imag_flags, eigvals, eigvecs",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "table", "summary", "arrays"],
        default="auto",
        help="Comparison output format",
    )

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

        if args.run_layout:
            required = {
                "--structure": args.structure,
                "--split": args.split,
                "--sweep_id": args.sweep_id,
                "--run_id": args.run_id,
            }

            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"--run-layout requires: {', '.join(missing)}"
                )

            print_run_layout(
                h5,
                structure=args.structure,
                split=args.split,
                sweep_id=args.sweep_id,
                run_id=args.run_id,
                print_values=args.values,
                max_full_size=args.max_full_size,
                preview_size=args.preview_size,
                precision=args.precision,
            )
            return


        if args.compare:
            required = {
                "--structure": args.structure,
                "--split": args.split,
                "--sweep_id": args.sweep_id,
                "--run_id": args.run_id,
            }

            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"--compare requires: {', '.join(missing)}"
                )

            compare_property(
                h5,
                structure=args.structure,
                split=args.split,
                sweep_id=args.sweep_id,
                run_id=args.run_id,
                prop=args.property,
                fmt=args.format,
            )
            return

        if args.details:
            for structure in selected:
                print("=" * 80)
                print(f"DETAILS: {structure}")
                print("=" * 80)

                path = f"structures/{structure}"
                if path not in h5:
                    print(f"[MISSING] {path}")
                    continue

                recurse_group(
                    h5[path],
                    indent="  ",
                    print_values=args.values,
                    max_full_size=args.max_full_size,
                    preview_size=args.preview_size,
                    precision=args.precision,
                )
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
