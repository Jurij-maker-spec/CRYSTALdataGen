 #!/usr/bin/env python3
"""
compare_distortion_cifs.py

Compare old vs newly generated distorted CIF folders.

Expected layout:

OLD_ROOT/
├── 000000/*.cif
├── 000001/*.cif
└── ...

NEW_ROOT/
├── 000000/*.cif
├── 000001/*.cif
└── ...

Writes:
    comparison_summary.json
    comparison_summary.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from ase.io import read


# ============================================================
# USER SETTINGS
# ============================================================

OLD_ROOT = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/structures/AlN/disto/sp_mixed_010_100_sc222/")
NEW_ROOT = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/structures/AlN/disto/sp_oldrandom_010_100_sc222/")

OUTPUT_DIR = NEW_ROOT / "comparison_old_vs_new"

CELL_ATOL = 1e-8
POS_ATOL = 1e-8
FRAC_ATOL = 1e-8

SKIP_MISSING = True
ALLOW_MULTIPLE_CIFS_USE_FIRST = True


# ============================================================
# HELPERS
# ============================================================

def find_cif(folder: Path) -> Path | None:
    cifs = sorted(folder.glob("*.cif"))

    if not cifs:
        return None

    if len(cifs) > 1:
        if not ALLOW_MULTIPLE_CIFS_USE_FIRST:
            raise RuntimeError(f"Multiple CIFs found in {folder}")
        print(f"[WARN] Multiple CIFs in {folder}, using {cifs[0].name}")

    return cifs[0]


def get_subdirs(root: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in sorted(root.iterdir())
        if p.is_dir()
    }


def minimum_image_frac_delta(frac_a: np.ndarray, frac_b: np.ndarray) -> np.ndarray:
    dfrac = frac_b - frac_a
    dfrac -= np.round(dfrac)
    return dfrac


def compare_one(old_cif: Path, new_cif: Path, distortion_id: str) -> dict:
    old_atoms = read(old_cif)
    new_atoms = read(new_cif)

    result = {
        "distortion_id": distortion_id,
        "old_cif": str(old_cif),
        "new_cif": str(new_cif),
        "status": "ok",
        "messages": [],
    }

    if len(old_atoms) != len(new_atoms):
        result["status"] = "fail"
        result["messages"].append(
            f"atom count mismatch old={len(old_atoms)} new={len(new_atoms)}"
        )
        return result

    old_symbols = old_atoms.get_chemical_symbols()
    new_symbols = new_atoms.get_chemical_symbols()

    if old_symbols != new_symbols:
        result["status"] = "fail"
        result["messages"].append("symbol/order mismatch")
        return result

    old_cell = old_atoms.cell.array
    new_cell = new_atoms.cell.array

    cell_diff = new_cell - old_cell
    max_cell_abs = float(np.max(np.abs(cell_diff)))

    old_pos = old_atoms.get_positions()
    new_pos = new_atoms.get_positions()

    pos_diff = new_pos - old_pos
    pos_norms = np.linalg.norm(pos_diff, axis=1)

    max_pos_abs = float(np.max(np.abs(pos_diff)))
    max_pos_norm = float(np.max(pos_norms))
    rms_pos = float(np.sqrt(np.mean(pos_norms**2)))

    old_frac = old_atoms.get_scaled_positions(wrap=False)
    new_frac = new_atoms.get_scaled_positions(wrap=False)

    frac_diff = minimum_image_frac_delta(old_frac, new_frac)
    frac_norms = np.linalg.norm(frac_diff, axis=1)

    max_frac_abs = float(np.max(np.abs(frac_diff)))
    max_frac_norm = float(np.max(frac_norms))
    rms_frac = float(np.sqrt(np.mean(frac_norms**2)))

    result.update({
        "n_atoms": len(old_atoms),
        "formula": old_atoms.get_chemical_formula(),

        "max_cell_abs_A": max_cell_abs,

        "max_pos_abs_A": max_pos_abs,
        "max_pos_norm_A": max_pos_norm,
        "rms_pos_A": rms_pos,

        "max_frac_abs": max_frac_abs,
        "max_frac_norm": max_frac_norm,
        "rms_frac": rms_frac,

        "cell_within_tol": bool(max_cell_abs <= CELL_ATOL),
        "pos_within_tol": bool(max_pos_abs <= POS_ATOL),
        "frac_within_tol": bool(max_frac_abs <= FRAC_ATOL),
    })

    if max_cell_abs > CELL_ATOL:
        result["status"] = "fail"
        result["messages"].append(
            f"cell mismatch max_abs={max_cell_abs:.3e} > {CELL_ATOL:.1e}"
        )

    if max_pos_abs > POS_ATOL:
        result["status"] = "fail"
        result["messages"].append(
            f"position mismatch max_abs={max_pos_abs:.3e} > {POS_ATOL:.1e}"
        )

    if max_frac_abs > FRAC_ATOL:
        result["status"] = "fail"
        result["messages"].append(
            f"fractional mismatch max_abs={max_frac_abs:.3e} > {FRAC_ATOL:.1e}"
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    if not OLD_ROOT.exists():
        raise FileNotFoundError(f"OLD_ROOT not found: {OLD_ROOT}")

    if not NEW_ROOT.exists():
        raise FileNotFoundError(f"NEW_ROOT not found: {NEW_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    old_dirs = get_subdirs(OLD_ROOT)
    new_dirs = get_subdirs(NEW_ROOT)

    common_ids = sorted(set(old_dirs) & set(new_dirs))
    only_old = sorted(set(old_dirs) - set(new_dirs))
    only_new = sorted(set(new_dirs) - set(old_dirs))

    results = []
    failed = []
    skipped = []

    print(f"[INFO] OLD_ROOT: {OLD_ROOT}")
    print(f"[INFO] NEW_ROOT: {NEW_ROOT}")
    print(f"[INFO] Common IDs: {len(common_ids)}")
    print(f"[INFO] Only old:   {len(only_old)}")
    print(f"[INFO] Only new:   {len(only_new)}")

    for distortion_id in common_ids:
        old_cif = find_cif(old_dirs[distortion_id])
        new_cif = find_cif(new_dirs[distortion_id])

        if old_cif is None or new_cif is None:
            msg = {
                "distortion_id": distortion_id,
                "status": "skipped",
                "old_cif_found": old_cif is not None,
                "new_cif_found": new_cif is not None,
            }

            if SKIP_MISSING:
                skipped.append(msg)
                print(f"[WARN] Skipping {distortion_id}: missing CIF")
                continue

            raise FileNotFoundError(f"Missing CIF for {distortion_id}: {msg}")

        result = compare_one(old_cif, new_cif, distortion_id)
        results.append(result)

        if result["status"] != "ok":
            failed.append(result)
            print(f"[FAIL] {distortion_id}: {'; '.join(result['messages'])}")

    n_ok = sum(r["status"] == "ok" for r in results)
    n_fail = sum(r["status"] == "fail" for r in results)

    summary = {
        "old_root": str(OLD_ROOT),
        "new_root": str(NEW_ROOT),
        "cell_atol": CELL_ATOL,
        "pos_atol": POS_ATOL,
        "frac_atol": FRAC_ATOL,

        "n_common": len(common_ids),
        "n_compared": len(results),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_skipped": len(skipped),

        "only_old": only_old,
        "only_new": only_new,
        "skipped": skipped,
        "results": results,
    }

    json_path = OUTPUT_DIR / "comparison_summary.json"
    csv_path = OUTPUT_DIR / "comparison_summary.csv"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = [
        "distortion_id",
        "status",
        "n_atoms",
        "formula",
        "max_cell_abs_A",
        "max_pos_abs_A",
        "max_pos_norm_A",
        "rms_pos_A",
        "max_frac_abs",
        "max_frac_norm",
        "rms_frac",
        "cell_within_tol",
        "pos_within_tol",
        "frac_within_tol",
        "messages",
        "old_cif",
        "new_cif",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["messages"] = " | ".join(r.get("messages", []))
            writer.writerow(row)

    print()
    print("[DONE]")
    print(f"Compared: {len(results)}")
    print(f"OK:       {n_ok}")
    print(f"Failed:   {n_fail}")
    print(f"Skipped:  {len(skipped)}")
    print(f"Wrote:    {json_path}")
    print(f"Wrote:    {csv_path}")


if __name__ == "__main__":
    main()
    