#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import h5py


DEFAULT_REF_DB = Path("data/ref_db.h5")


def is_pbe_split(split_name: str) -> bool:
    return "_PBE" in str(split_name)


def pbe_structure_for(base_structure: str) -> str:
    base_structure = str(base_structure)

    if base_structure.endswith("_PBE"):
        return base_structure

    return base_structure + "_PBE"


def count_runs_in_split(split_group: h5py.Group) -> dict[str, int]:
    """
    Return:
        {sweep_id: number_of_direct_run_groups}
    """
    out = {}

    for sweep_id, sweep_group in split_group.items():
        if not isinstance(sweep_group, h5py.Group):
            continue

        n_runs = sum(
            1
            for _, obj in sweep_group.items()
            if isinstance(obj, h5py.Group)
        )

        out[str(sweep_id)] = n_runs

    return out


def split_signature(h5: h5py.File, structure: str, split: str) -> dict[str, int] | None:
    path = f"structures/{structure}/evaluations/{split}"

    if path not in h5:
        return None

    return count_runs_in_split(h5[path])


def signatures_equal(sig_a: dict[str, int] | None, sig_b: dict[str, int] | None) -> bool:
    if sig_a is None or sig_b is None:
        return False

    return sig_a == sig_b


def find_misplaced_pbe_splits(
    h5: h5py.File,
    *,
    structure_filter: str | None = None,
):
    """
    Find PBE evaluation splits stored below non-PBE first-level structure groups.

    Example:
        WRONG:
            /structures/Al2O3/evaluations/Al2O3_PBE

        EXPECTED COPY:
            /structures/Al2O3_PBE/evaluations/Al2O3_PBE

    Only returns candidates where the target PBE structure exists and has
    an identical split signature.
    """
    if "structures" not in h5:
        raise KeyError("Missing /structures group")

    structures_root = h5["structures"]

    candidates = []

    for structure_name in sorted(structures_root.keys()):
        if structure_filter is not None and structure_name != structure_filter:
            continue

        if structure_name.endswith("_PBE"):
            continue

        eval_root_path = f"structures/{structure_name}/evaluations"

        if eval_root_path not in h5:
            continue

        eval_root = h5[eval_root_path]

        for split_name in sorted(eval_root.keys()):
            if not is_pbe_split(split_name):
                continue

            source_path = f"structures/{structure_name}/evaluations/{split_name}"
            target_structure = pbe_structure_for(structure_name)
            target_path = f"structures/{target_structure}/evaluations/{split_name}"

            source_sig = split_signature(h5, structure_name, split_name)
            target_sig = split_signature(h5, target_structure, split_name)

            safe_to_delete = signatures_equal(source_sig, target_sig)

            candidates.append({
                "structure": structure_name,
                "split": split_name,
                "source_path": source_path,
                "target_structure": target_structure,
                "target_path": target_path,
                "source_sig": source_sig,
                "target_sig": target_sig,
                "safe_to_delete": safe_to_delete,
            })

    return candidates


def delete_misplaced_pbe_splits(
    ref_db_path: str | Path,
    *,
    structure_filter: str | None = None,
    apply: bool = False,
    force: bool = False,
):
    ref_db_path = Path(ref_db_path)

    if not ref_db_path.exists():
        raise FileNotFoundError(ref_db_path)

    mode = "APPLY" if apply else "DRY-RUN"

    print("=" * 100)
    print(f"DELETE MISPLACED PBE EVALUATION SPLITS [{mode}]")
    print("=" * 100)
    print(f"DB               : {ref_db_path}")
    print(f"structure filter : {structure_filter}")
    print(f"force            : {force}")
    print()

    n_candidates = 0
    n_safe = 0
    n_deleted = 0
    n_blocked = 0

    with h5py.File(ref_db_path, "a" if apply else "r") as h5:
        candidates = find_misplaced_pbe_splits(
            h5,
            structure_filter=structure_filter,
        )

        if not candidates:
            print("No misplaced PBE splits found.")
            return

        for c in candidates:
            n_candidates += 1

            print("-" * 100)
            print(f"source : {c['source_path']}")
            print(f"target : {c['target_path']}")
            print(f"source signature : {c['source_sig']}")
            print(f"target signature : {c['target_sig']}")

            if c["safe_to_delete"]:
                n_safe += 1
                print("status : SAFE, identical split/sweep/run-count signature exists under target")
            else:
                n_blocked += 1
                print("status : BLOCKED, no identical target signature")

            should_delete = c["safe_to_delete"] or force

            if not should_delete:
                continue

            if not apply:
                print("action : would delete")
                continue

            del h5[c["source_path"]]
            n_deleted += 1
            print("action : deleted")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"candidates : {n_candidates}")
    print(f"safe       : {n_safe}")
    print(f"blocked    : {n_blocked}")
    print(f"deleted    : {n_deleted}")

    if not apply:
        print()
        print("Dry-run only. Re-run with --apply to delete safe candidates.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Delete misplaced PBE evaluation splits from non-PBE structure groups, "
            "but only if identical split/sweep/run-count copies exist under the "
            "corresponding _PBE structure."
        )
    )
    parser.add_argument(
        "--ref_db",
        type=Path,
        default=DEFAULT_REF_DB,
        help="Path to ref_db.h5.",
    )
    parser.add_argument(
        "--structure",
        default=None,
        help="Optional base structure filter, e.g. SiO2, AlN, Al2O3.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this, dry-run only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete even if target signature is missing/different. "
            "Use only after manual inspection."
        ),
    )

    args = parser.parse_args()

    delete_misplaced_pbe_splits(
        args.ref_db,
        structure_filter=args.structure,
        apply=args.apply,
        force=args.force,
    )


if __name__ == "__main__":
    main()
    