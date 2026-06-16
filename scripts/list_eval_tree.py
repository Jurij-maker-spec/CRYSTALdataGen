#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import h5py


DEFAULT_REF_DB = Path("data/ref_db.h5")


def count_run_groups(sweep_group: h5py.Group) -> int:
    """
    Count direct run groups below one sweep_id group.
    """
    return sum(
        1
        for _, obj in sweep_group.items()
        if isinstance(obj, h5py.Group)
    )


def list_eval_tree(
    ref_db_path: str | Path,
    *,
    structure_filter: str | None = None,
    show_empty: bool = False,
):
    ref_db_path = Path(ref_db_path)

    if not ref_db_path.exists():
        raise FileNotFoundError(ref_db_path)

    with h5py.File(ref_db_path, "r") as h5:
        if "structures" not in h5:
            raise KeyError("Missing top-level group: /structures")

        structures_root = h5["structures"]

        print("=" * 100)
        print("REF DB EVALUATION TREE")
        print("=" * 100)
        print(f"DB: {ref_db_path}")
        print()

        total_structures = 0
        total_splits = 0
        total_sweeps = 0
        total_runs = 0

        for structure_name in sorted(structures_root.keys()):
            if structure_filter is not None and structure_name != structure_filter:
                continue

            structure_group = structures_root[structure_name]
            if not isinstance(structure_group, h5py.Group):
                continue

            eval_path = f"structures/{structure_name}/evaluations"

            if eval_path not in h5:
                if show_empty:
                    print(f"{structure_name}/")
                    print("  evaluations: MISSING")
                continue

            eval_root = h5[eval_path]
            total_structures += 1

            print(f"{structure_name}/")
            print("  evaluations/")

            structure_run_count = 0
            structure_sweep_count = 0
            structure_split_count = 0

            for split_name in sorted(eval_root.keys()):
                split_group = eval_root[split_name]
                if not isinstance(split_group, h5py.Group):
                    continue

                structure_split_count += 1
                total_splits += 1

                print(f"    {split_name}/")

                split_run_count = 0
                split_sweep_count = 0

                for sweep_id in sorted(split_group.keys()):
                    sweep_group = split_group[sweep_id]
                    if not isinstance(sweep_group, h5py.Group):
                        continue

                    n_runs = count_run_groups(sweep_group)

                    split_sweep_count += 1
                    structure_sweep_count += 1
                    total_sweeps += 1

                    split_run_count += n_runs
                    structure_run_count += n_runs
                    total_runs += n_runs

                    print(f"      {sweep_id}: {n_runs} runs")

                print(
                    f"      split total: "
                    f"{split_sweep_count} sweeps, {split_run_count} runs"
                )

            print(
                f"  structure total: "
                f"{structure_split_count} splits, "
                f"{structure_sweep_count} sweeps, "
                f"{structure_run_count} runs"
            )
            print()

        print("=" * 100)
        print("TOTAL")
        print("=" * 100)
        print(f"structures with evaluations : {total_structures}")
        print(f"splits                      : {total_splits}")
        print(f"sweeps                      : {total_sweeps}")
        print(f"runs                        : {total_runs}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "List ref_db evaluation hierarchy up to sweep_id level. "
            "For each sweep_id, print only the number of runs."
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
        help="Optional structure filter, e.g. SiO2 or SiO2_PBE.",
    )
    parser.add_argument(
        "--show_empty",
        action="store_true",
        help="Also show structures without an evaluations group.",
    )

    args = parser.parse_args()

    list_eval_tree(
        args.ref_db,
        structure_filter=args.structure,
        show_empty=args.show_empty,
    )


if __name__ == "__main__":
    main()
    