#!/usr/bin/env python3

from pathlib import Path
import argparse

RESULTS_ROOT = Path("results")

DELETE_SUFFIXES = (
    "_ir_corr_comp.png",
    "_ir_comparison.png",
)


def is_inside_pbe_directory(path: Path) -> bool:
    """
    Only touch files somewhere inside directories containing 'PBE'.
    """
    return any("PBE" in part for part in path.parts)


def should_delete(path: Path) -> bool:
    name = path.name

    # only target IR comparison plots
    if not any(name.endswith(suffix) for suffix in DELETE_SUFFIXES):
        return False

    # ONLY operate inside PBE result directories
    if not is_inside_pbe_directory(path):
        return False

    # keep proper PBE plots
    if "PBE" in name:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Delete wrongly generated non-PBE IR plots inside PBE result folders."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be deleted.",
    )

    args = parser.parse_args()

    to_delete = []

    for path in RESULTS_ROOT.rglob("*.png"):
        if should_delete(path):
            to_delete.append(path)

    print("\n========================================")
    print(f"Found {len(to_delete)} plots to delete")
    print("========================================\n")

    for path in sorted(to_delete):
        print(path)

    if args.dry_run:
        print("\n[DRY RUN] No files deleted.")
        return

    deleted = 0
    failed = 0

    print("\nDeleting files...\n")

    for path in to_delete:
        try:
            path.unlink()
            deleted += 1
            print(f"Deleted: {path}")
        except Exception as exc:
            failed += 1
            print(f"Failed : {path}")
            print(f"Reason : {exc}")

    print("\n========================================")
    print(f"Deleted : {deleted}")
    print(f"Failed  : {failed}")
    print("========================================")


if __name__ == "__main__":
    main()
    