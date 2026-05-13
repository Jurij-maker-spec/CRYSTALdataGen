#!/usr/bin/env python3
from pathlib import Path
import argparse
import h5py
import numpy as np


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


def check_structure(h5, structure):
    path = f"structures/{structure}"
    if path not in h5:
        print(f"[MISSING] {path}")
        return False

    sg = h5[path]
    print("=" * 80)
    print(f"STRUCTURE: {structure}")
    print("=" * 80)

    if "crystal" not in sg:
        print("[MISSING] crystal group")
        return False

    cg = sg["crystal"]
    recurse_group(cg, indent="  ")

    required = [
        "geometry/atomic_numbers",
        "geometry/positions_A",
        "geometry/cell_A",
        "hessian_cart_eV_A2",
        "hessian_mw_SI",
        "eigvals_SI",
        "eigvecs_mw",
        "frequencies_cm1",
        "imag_flags",
    ]

    ok = True
    print("\nRequired dataset check:")
    for rel in required:
        exists = rel in cg
        print(f"  {'OK' if exists else 'MISSING'}  crystal/{rel}")
        ok = ok and exists

    if "geometry/atomic_numbers" in cg and "hessian_cart_eV_A2" in cg:
        nat = len(cg["geometry/atomic_numbers"])
        expected = 3 * nat
        H = cg["hessian_cart_eV_A2"]
        print("\nShape consistency:")
        print(f"  natoms = {nat}")
        print(f"  expected Hessian shape = ({expected}, {expected})")
        print(f"  actual Hessian shape   = {H.shape}")
        if H.shape != (expected, expected):
            ok = False
            print("  [FAIL] Hessian shape mismatch")
        else:
            print("  [OK] Hessian shape")

    if "frequencies_cm1" in cg:
        freqs = cg["frequencies_cm1"][()]
        print("\nFrequency summary:")
        print(f"  n_modes = {len(freqs)}")
        print(f"  min     = {np.min(freqs):.6f} cm^-1")
        print(f"  max     = {np.max(freqs):.6f} cm^-1")
        print(f"  first 10: {freqs[:10]}")

    if "imag_flags" in cg:
        imag = cg["imag_flags"][()]
        print(f"  n_imag  = {int(np.sum(imag))}")

    if "born_charges" in cg:
        bec = cg["born_charges"][()]
        print("\nBEC summary:")
        print(f"  shape = {bec.shape}")
        print(f"  acoustic sum Σ_i Z*_i:")
        print(np.sum(bec, axis=0))

    if "dielectric_tensor" in cg:
        print("\nDielectric tensor:")
        print(cg["dielectric_tensor"][()])

    if "evaluations" in sg:
        print("\nEvaluations:")
        for run_id, eg in sg["evaluations"].items():
            print(f"\n  RUN: {run_id}")
            recurse_group(eg, indent="--    ")

    #if "evaluations" in sg:
    #    print("\nEvaluations:")
    #    for run_id in sg["evaluations"].keys():
    #        print(f"  - {run_id}")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Inspect and sanity-check ref_db.h5.")
    parser.add_argument("ref_db", type=Path, help="Path to ref_db.h5")
    parser.add_argument("--structure", default=None, help="Structure name to inspect")
    parser.add_argument("--list", action="store_true", help="Only list structures")
    args = parser.parse_args()

    if not args.ref_db.exists():
        raise FileNotFoundError(args.ref_db)

    with h5py.File(args.ref_db, "r") as h5:
        if "structures" not in h5:
            raise KeyError("Missing top-level group: /structures")

        structures = sorted(h5["structures"].keys())

        print(f"DB: {args.ref_db}")
        print(f"Structures: {len(structures)}")
        for s in structures:
            print(f"  - {s}")

        if args.list:
            return

        if args.structure is not None:
            if args.structure not in structures:
                raise KeyError(f"Structure not found: {args.structure}")
            ok = check_structure(h5, args.structure)
        else:
            ok = True
            for structure in structures:
                ok = check_structure(h5, structure) and ok

        print("\n" + "=" * 80)
        print("RESULT:", "OK" if ok else "CHECK WARNINGS / FAILURES")


if __name__ == "__main__":
    main()
