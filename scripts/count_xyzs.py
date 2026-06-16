#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
from ase.io import iread


def count_ase_frames(path: Path) -> int:
    """
    Count structures/frames in an ASE-readable trajectory file.

    Works for:
      - extxyz
      - xyz
      - many ASE-supported formats

    Does not load all structures into memory.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    n = 0
    for _ in iread(path, index=":"):
        n += 1

    return n


def count_pair(train_file: Path, valid_file: Path | None = None) -> dict:
    train_n = count_ase_frames(train_file)

    out = {
        "train_file": train_file,
        "train_size": train_n,
    }

    if valid_file is not None:
        valid_n = count_ase_frames(valid_file)
        out["valid_file"] = valid_file
        out["valid_size"] = valid_n
        out["total_size"] = train_n + valid_n

        if train_n + valid_n > 0:
            out["train_fraction"] = train_n / (train_n + valid_n)
            out["valid_fraction"] = valid_n / (train_n + valid_n)

    return out


def print_human(result: dict):
    print("=" * 80)
    print("XYZ STRUCTURE COUNT")
    print("=" * 80)

    print(f"train_file : {result['train_file']}")
    print(f"train_size : {result['train_size']}")

    if "valid_file" in result:
        print(f"valid_file : {result['valid_file']}")
        print(f"valid_size : {result['valid_size']}")
        print(f"total_size : {result['total_size']}")
        print(f"train_frac : {result['train_fraction']:.4f}")
        print(f"valid_frac : {result['valid_fraction']:.4f}")


def print_tsv(result: dict):
    header = ["train_file", "train_size"]

    if "valid_file" in result:
        header += [
            "valid_file",
            "valid_size",
            "total_size",
            "train_fraction",
            "valid_fraction",
        ]

    print("\t".join(header))

    row = [
        str(result["train_file"]),
        str(result["train_size"]),
    ]

    if "valid_file" in result:
        row += [
            str(result["valid_file"]),
            str(result["valid_size"]),
            str(result["total_size"]),
            f"{result['train_fraction']:.8f}",
            f"{result['valid_fraction']:.8f}",
        ]

    print("\t".join(row))


def main():
    parser = argparse.ArgumentParser(
        description="Count structures in train.xyz and valid.xyz files."
    )
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Path to train.xyz / train.extxyz.",
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=None,
        help="Optional path to valid.xyz / valid.extxyz.",
    )
    parser.add_argument(
        "--format",
        choices=["human", "tsv"],
        default="human",
        help="Output format.",
    )

    args = parser.parse_args()

    result = count_pair(args.train, args.valid)

    if args.format == "human":
        print_human(result)
    else:
        print_tsv(result)


if __name__ == "__main__":
    main()
    