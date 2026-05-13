#!/usr/bin/env python3
"""
extract_distortions_template_script.py

Extract reusable distortion templates from:

ROOT/structures/STRUCT/disto/sp_mixed_XXX_YYY/
├── 000000/
│   └── *.cif
├── 000001/
│   └── *.cif
└── ...

Reference is assumed to be outside this folder.

Output:

ROOT/structures/STRUCT/disto/sp_mixed_XXX_YYY/distortion_templates/
├── distortion_templates.json
└── distortion_vectors.npz
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
from ase.io import read

# ============================================================
# USER SETTINGS
# ============================================================
ROOT = Path('/home/jha/jha/python_scripts/CRYSTALdataGen/')
# -----------------------------
# SiO2 PBE 010
# -----------------------------
# STRUCT = 'SiO2'
# REFERENCE_CIF = ROOT / 'structures/SiO2/SiO2_geoopt_clean.cif'
# DISTORTION_BATCH_DIR = ROOT / 'structures/SiO2/disto/sp_mixed_010_100'

# -----------------------------
# TiO2 PBE 010
# -----------------------------
# STRUCT = 'TiO2_rutil'
# REFERENCE_CIF = ROOT / 'structures/TiO2_rutil/TiO2_rutil_geoopt_clean.cif'
# DISTORTION_BATCH_DIR = ROOT / 'structures/TiO2_rutil/disto/sp_mixed_010_100'

# -----------------------------
# Al2O3 PBE 010
# -----------------------------
# STRUCT = 'Al2O3'
# REFERENCE_CIF = ROOT / 'structures/Al2O3/Al2O3_geoopt_clean.cif'
# DISTORTION_BATCH_DIR = ROOT / 'structures/Al2O3/disto/sp_mixed_010_100'


# -----------------------------
# AlN PBE 000
# -----------------------------
STRUCT = 'AlN'
REFERENCE_CIF = ROOT / 'structures/AlN/AlN_geoopt_clean.cif'
DISTORTION_BATCH_DIR = ROOT / 'structures/AlN/disto/sp_mixed_000_150'


OUTPUT_DIR = ROOT / 'configs/disto_cfg/disto_arrays'

SKIP_BAD = False
REQUIRE_SAME_CELL = False
SAVE_CELL_DEFORMATION = True
CELL_ATOL = 1e-6


# ============================================================
# HELPERS
# ============================================================

def resolve_reference():
    if REFERENCE_CIF is not None:
        return Path(REFERENCE_CIF)

    raise FileNotFoundError("Could not locate reference CIF.")


def find_cif(folder: Path):
    files = sorted(folder.glob("*.cif"))
    if not files:
        return None
    if len(files) > 1:
        print(f"[WARN] Multiple CIFs in {folder}, using {files[0].name}")
    return files[0]


def get_distortion_dirs():
    return sorted(p for p in DISTORTION_BATCH_DIR.iterdir() if p.is_dir())


def mic_frac_delta(ref_frac, dist_frac):
    d = dist_frac - ref_frac
    d -= np.round(d)
    return d


def check(ref_atoms, dist_atoms, dist_id):
    if len(ref_atoms) != len(dist_atoms):
        raise ValueError(
            f"{dist_id}: atom count mismatch "
            f"ref={len(ref_atoms)}, dist={len(dist_atoms)}"
        )

    if ref_atoms.get_chemical_symbols() != dist_atoms.get_chemical_symbols():
        raise ValueError(f"{dist_id}: symbol/order mismatch")

    if REQUIRE_SAME_CELL:
        if not np.allclose(
            ref_atoms.cell.array,
            dist_atoms.cell.array,
            atol=CELL_ATOL,
        ):
            raise ValueError(f"{dist_id}: cell mismatch")


def extract(ref_atoms, dist_atoms, dist_id):
    """
    Extract distortion as:

    1. fractional internal atomic displacement
    2. Cartesian internal displacement
    3. cell deformation matrix F, where:

           dist_cell = ref_cell @ F

    ASE uses row-vector cells, so positions are:

           cart = frac @ cell
    """

    check(ref_atoms, dist_atoms, dist_id)

    ref_cell = ref_atoms.cell.array
    dist_cell = dist_atoms.cell.array

    ref_frac = ref_atoms.get_scaled_positions(wrap=False)
    dist_frac = dist_atoms.get_scaled_positions(wrap=False)

    # Internal atomic displacement in fractional coordinates
    dfrac = dist_frac - ref_frac
    dfrac -= np.round(dfrac)

    # Cartesian displacement inside the distorted cell
    cart_internal_disp_A = dfrac @ dist_cell

    # Cell deformation:
    # dist_cell = ref_cell @ F
    cell_deformation = np.linalg.solve(ref_cell, dist_cell)

    norms = np.linalg.norm(cart_internal_disp_A, axis=1)

    return (
        dfrac,
        cart_internal_disp_A,
        cell_deformation,
        dist_cell,
        float(np.sqrt(np.mean(norms**2))),
        float(np.mean(norms)),
        float(np.max(norms)),
    )

# ============================================================
# MAIN
# ============================================================

def main():
    if not DISTORTION_BATCH_DIR.exists():
        raise FileNotFoundError(DISTORTION_BATCH_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_cif = resolve_reference()
    ref_atoms = read(reference_cif)

    print(f"[INFO] Batch dir: {DISTORTION_BATCH_DIR}")
    print(f"[INFO] Reference: {reference_cif}")

    ids = []
    frac_all = []
    cart_all = []
    cell_deformation_all = []   
    dist_cell_all = []
    rms_all = []
    mean_all = []
    max_all = []

    meta = {}
    failed = {}

    for d in get_distortion_dirs():
        dist_id = d.name

        try:
            cif = find_cif(d)
            if cif is None:
                raise RuntimeError("No CIF found")

            dist_atoms = read(cif)

            (
                dfrac,
                cart_internal,
                cell_deformation,
                dist_cell,
                rms,
                mean,
                maxv,
            ) = extract(ref_atoms, dist_atoms, dist_id)

            ids.append(dist_id)
            frac_all.append(dfrac)
            cart_all.append(cart_internal)
            cell_deformation_all.append(cell_deformation)
            dist_cell_all.append(dist_cell)

            rms_all.append(rms)
            mean_all.append(mean)
            max_all.append(maxv)

            meta[dist_id] = {
                "source": str(cif.relative_to(DISTORTION_BATCH_DIR)),
                "rms_A": rms,
                "mean_A": mean,
                "max_A": maxv,
                "has_cell_deformation": True,
            }

        except Exception as e:
            if SKIP_BAD:
                print(f"[WARN] {dist_id}: {e}")
                failed[dist_id] = str(e)
                continue
            raise

    if not ids:
        raise RuntimeError("No valid distortions extracted")

    frac_all = np.array(frac_all)
    cart_all = np.array(cart_all)
    cell_deformation_all = np.array(cell_deformation_all)
    dist_cell_all = np.array(dist_cell_all)

    ids = np.array(ids, dtype=object)

    rms_all = np.array(rms_all)
    mean_all = np.array(mean_all)
    max_all = np.array(max_all)

    npz_path = OUTPUT_DIR / f"disvec_{STRUCT}.npz"
    json_path = OUTPUT_DIR / f"dis_templates_{STRUCT}.json"

    np.savez_compressed(
        npz_path,
        ids=ids,
        frac_disp=frac_all,
        cart_internal_disp_A=cart_all,
        rms_A=rms_all,
        mean_A=mean_all,
        max_A=max_all,

        symbols=np.array(ref_atoms.get_chemical_symbols(), dtype=object),

        reference_cell_A=ref_atoms.cell.array,
        distorted_cells_A=dist_cell_all,
        cell_deformation=cell_deformation_all,
    )

    metadata = {
        "created": datetime.now().isoformat(),
        "batch_dir": str(DISTORTION_BATCH_DIR),
        "reference": str(reference_cif),
        "n_distortions": len(ids),
        "n_failed": len(failed),

        "assumptions": {
            "no_supercells": True,
            "same_atom_order_required": True,
            "same_atom_count_required": True,
            "same_cell_required": REQUIRE_SAME_CELL,
            "cell_deformation_saved": SAVE_CELL_DEFORMATION,
            "cell_relation": "distorted_cell = reference_cell @ cell_deformation",
        },

        "stats": {
            "rms_min": float(rms_all.min()),
            "rms_mean": float(rms_all.mean()),
            "rms_max": float(rms_all.max()),
        },

        "distortions": meta,
        "failed": failed,
        "coordinate_convention": {
            "frac_disp": (
                "Internal atomic displacement in fractional coordinates, "
                "wrapped by minimum-image convention."
            ),
            "cart_internal_disp_A": (
                "Internal atomic displacement in Angstrom, computed as "
                "frac_disp @ distorted_cell."
            ),
            "cell_deformation": (
                "3x3 deformation matrix F with distorted_cell = reference_cell @ F."
            ),
            "recommended_reuse": (
                "For full reproduction: new_cell_distorted = new_cell @ F, "
                "new_scaled_positions = new_ref_scaled_positions + frac_disp."
            ),
        },
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[DONE]")
    print(f"Valid: {len(ids)}, Failed: {len(failed)}")
    print(f"Wrote: {npz_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
