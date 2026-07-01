#!/usr/bin/env python3

import h5py
import numpy as np
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent

H5_PATH = ROOT / "../data/train_db.h5"


def decode_if_bytes(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x


def print_attrs(obj, indent="  "):
    if len(obj.attrs) == 0:
        return
    print(f"{indent}Attributes:")
    for k, v in obj.attrs.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind == "S":
                v = [decode_if_bytes(i) for i in v]
            print(f"{indent}  {k}: array(shape={v.shape}, dtype={v.dtype})")
        else:
            print(f"{indent}  {k}: {decode_if_bytes(v)}")


def print_dataset_preview(name, dset, indent="  ", max_items=6):
    arr = dset[()]
    shape = dset.shape
    dtype = dset.dtype

    print(f"{indent}{name}: shape={shape}, dtype={dtype}")

    if np.isscalar(arr):
        print(f"{indent}  value={arr}")
        return

    arr = np.asarray(arr)

    if arr.dtype.kind == "S":
        flat = arr.reshape(-1)
        preview = [decode_if_bytes(x) for x in flat[:max_items]]
        suffix = " ..." if flat.size > max_items else ""
        print(f"{indent}  preview={preview}{suffix}")
        return

    if arr.ndim == 1:
        preview = arr[:max_items]
        suffix = " ..." if arr.size > max_items else ""
        print(f"{indent}  preview={preview}{suffix}")
    elif arr.ndim >= 2:
        preview = arr[: min(arr.shape[0], 3)]
        print(f"{indent}  preview=\n{preview}")


def print_common_structure_group(g, title):
    print(f"\n=== {title} ===")

    for key in ["positions", "atomic_numbers", "lattice", "dft_forces", "stress", "energy"]:
        if key in g:
            print_dataset_preview(key, g[key], indent="  ")

    print_attrs(g, indent="  ")
    print("=" * (len(title) + 8))


def print_primitive_group(g, material):
    print(f"\n=== PRIMITIVE REFERENCE: {material} ===")

    keys_in_order = [
        "positions",
        "atomic_numbers",
        "lattice",
        "born_charges",
        "born_species",
        "optical_phonon_frequencies",
        "all_phonon_frequencies",
        "intensities",
        "degeneracies",
        "irreps",
    ]

    for key in keys_in_order:
        if key in g:
            print_dataset_preview(key, g[key], indent="  ")

    print_attrs(g, indent="  ")
    print("=" * (len(material) + 31))


def print_distortion(h5, material, distortion_id):
    mat = h5["structures"][material]

    if "distortions" not in mat:
        print(f"No distortions group found for {material}.")
        return

    if distortion_id not in mat["distortions"]:
        print(f"Distortion {distortion_id} not found for {material}.")
        return

    g = mat["distortions"][distortion_id]
    print_common_structure_group(g, f"DISTORTION {material}/{distortion_id}")


def print_material_summary(material, g):
    print(f"\n--- {material} ---")

    n_found = g.attrs.get("n_singlepoints_found", None)
    n_written = g.attrs.get("n_distortions_written", None)
    n_failed = g.attrs.get("n_distortions_failed", None)

    if "reference" in g:
        ref = g["reference"]
        n_atoms = ref["positions"].shape[0] if "positions" in ref else "?"
        energy = ref["energy"][()] if "energy" in ref else "?"
        print(f"  reference present           : yes")
        print(f"  reference atoms            : {n_atoms}")
        print(f"  reference energy [eV]      : {energy}")
    else:
        print(f"  reference present           : no")

    if "primitive_reference" in g:
        prim = g["primitive_reference"]
        print("  primitive_reference present: yes")

        if "all_phonon_frequencies" in prim:
            nfreq = prim["all_phonon_frequencies"].shape[0]
            print(f"  phonon frequencies         : {nfreq}")
        if "born_charges" in prim:
            print(f"  born charges shape         : {prim['born_charges'].shape}")
    else:
        print("  primitive_reference present: no")

    ndist = len(g["distortions"]) if "distortions" in g else 0
    nfail = len(g["failed_distortions"]) if "failed_distortions" in g else 0

    print(f"  distortions written        : {ndist}")
    print(f"  failed distortions         : {nfail}")

    if n_found is not None:
        print(f"  singlepoints found         : {n_found}")
    if n_written is not None:
        print(f"  attr written               : {n_written}")
    if n_failed is not None:
        print(f"  attr failed                : {n_failed}")

    if n_found is not None and n_written is not None and n_failed is not None:
        if n_found != (n_written + n_failed):
            print("  WARNING                    : found != written + failed")


def print_summary(h5):
    print("\n=== DATASET SUMMARY ===")

    if "structures" not in h5:
        print("No structures group found.")
        return

    struct_group = h5["structures"]
    materials = list(struct_group.keys())

    print(f"Number of materials: {len(materials)}")
    print(f"Materials: {', '.join(materials) if materials else '(none)'}")

    total_distortions = 0
    n_refs = 0
    n_prims = 0

    for material in materials:
        g = struct_group[material]
        if "reference" in g:
            n_refs += 1
        if "primitive_reference" in g:
            n_prims += 1
        if "distortions" in g:
            total_distortions += len(g["distortions"])

    print(f"Materials with reference           : {n_refs}")
    print(f"Materials with primitive_reference : {n_prims}")
    print(f"Total distortions                  : {total_distortions}")

    if materials:
        first_key = materials[0]
        print_material_summary(first_key, struct_group[first_key])

    if "atomic_energies" in h5:
        zs = list(h5["atomic_energies"].keys())
        print(f"\nAtomic energies present for Z: {', '.join(zs) if zs else '(none)'}")
    else:
        print("\nNo atomic_energies group found.")

    print("========================\n")


def print_failure_summary(h5):
    if "structures" not in h5:
        print("No structures group found.")
        return

    print("\n=== FAILURE SUMMARY ===")
    total_failed = 0

    for material, g in h5["structures"].items():
        n_found = g.attrs.get("n_singlepoints_found", 0)
        n_written = g.attrs.get("n_distortions_written", 0)
        n_failed = g.attrs.get("n_distortions_failed", 0)
        total_failed += n_failed

        print(
            f"{material:12s} | found={int(n_found):4d} | written={int(n_written):4d} | failed={int(n_failed):4d}"
        )

    print(f"\nTotal failed distortions: {total_failed}")
    print("=========================\n")


def print_failed_distortions(h5, material):
    mat = h5["structures"][material]

    if "failed_distortions" not in mat:
        print(f"No failed_distortions group found for {material}.")
        return

    failed = mat["failed_distortions"]
    keys = list(failed.keys())

    print(f"\n=== FAILED DISTORTIONS: {material} ===")
    print(f"Count: {len(keys)}")

    if not keys:
        print("No failed distortions.")
        print("=====================================\n")
        return

    for key in keys:
        g = failed[key]

        source_file = decode_if_bytes(g["source_file"][()])
        source_name = decode_if_bytes(g["source_name"][()])
        reason = decode_if_bytes(g["reason"][()])
        error_message = decode_if_bytes(g["error_message"][()])

        print(f"\n{key}")
        print(f"  source_name  : {source_name}")
        print(f"  source_file  : {source_file}")
        print(f"  reason       : {reason}")
        if error_message:
            print(f"  error_message: {error_message}")

    print("=====================================\n")


def print_material(h5, material):
    if "structures" not in h5 or material not in h5["structures"]:
        print(f"Material {material} not found.")
        return

    g = h5["structures"][material]
    print_material_summary(material, g)

    if "reference" in g:
        print_common_structure_group(g["reference"], f"REFERENCE {material}")

    if "primitive_reference" in g:
        print_primitive_group(g["primitive_reference"], material)

    if "distortions" in g:
        dist_keys = list(g["distortions"].keys())
        print(f"\nFirst distortions for {material}: {dist_keys[:10]}")
        if len(dist_keys) > 10:
            print("...")


def print_all_materials(h5):
    if "structures" not in h5:
        print("No structures group found.")
        return

    for material in h5["structures"].keys():
        print_material(h5, material)


def print_atomic_energies(h5):
    if "atomic_energies" not in h5:
        print("No atomic_energies group found.")
        return

    print("\n=== ATOMIC ENERGIES ===")
    for z in h5["atomic_energies"].keys():
        g = h5["atomic_energies"][z]
        print(f"\nZ = {z}")
        for key in g.keys():
            print_dataset_preview(key, g[key], indent="  ")
        print_attrs(g, indent="  ")
    print("=======================\n")


def print_help():
    print(
        """
Usage:
  python inspect_hdf5.py
      Print global summary

  python inspect_hdf5.py all
      Print all materials

  python inspect_hdf5.py atomic
      Print atomic energies

  python inspect_hdf5.py <material>
      Print one material, e.g.
      python inspect_hdf5.py AlN

  python inspect_hdf5.py <material> primitive
      Print primitive_reference of one material

  python inspect_hdf5.py <material> reference
      Print reference of one material

  python inspect_hdf5.py <material> distortions
      List distortion ids of one material

  python inspect_hdf5.py <material> distortion <id>
      Print one distortion, e.g.
      python inspect_hdf5.py AlN distortion 000000

  python inspect_hdf5.py failures
      Print failure summary for all materials

  python inspect_hdf5.py <material> failures
      Print all failed distortions of one material
"""
    )


def main():
    if not H5_PATH.exists():
        print(f"HDF5 file not found: {H5_PATH}")
        return

    with h5py.File(H5_PATH, "r") as h5:
        args = sys.argv[1:]

        if len(args) == 0:
            print_summary(h5)
            return

        if args[0] == "help":
            print_help()
            return

        if args[0] == "all":
            print_all_materials(h5)
            return

        if args[0] == "atomic":
            print_atomic_energies(h5)
            return

        if args[0] == "failures":
            print_failure_summary(h5)
            return

        material = args[0]

        if "structures" not in h5 or material not in h5["structures"]:
            print(f"Unknown material: {material}")
            print_help()
            return

        mat_group = h5["structures"][material]

        if len(args) == 1:
            print_material(h5, material)
            return

        cmd = args[1]

        if cmd == "primitive":
            if "primitive_reference" in mat_group:
                print_primitive_group(mat_group["primitive_reference"], material)
            else:
                print(f"No primitive_reference found for {material}.")
            return

        if cmd == "reference":
            if "reference" in mat_group:
                print_common_structure_group(mat_group["reference"], f"REFERENCE {material}")
            else:
                print(f"No reference found for {material}.")
            return

        if cmd == "distortions":
            if "distortions" not in mat_group:
                print(f"No distortions found for {material}.")
                return
            keys = list(mat_group["distortions"].keys())
            print(f"\n=== DISTORTIONS: {material} ===")
            print(f"Count: {len(keys)}")
            prt_max = 250
            for k in keys[:prt_max]:
                print(k)
            if len(keys) > prt_max:
                print("...")
            print("============================\n")
            return

        if cmd == "distortion":
            if len(args) < 3:
                print("Missing distortion id.")
                print_help()
                return
            distortion_id = args[2]
            print_distortion(h5, material, distortion_id)
            return

        if cmd == "failures":
            print_failed_distortions(h5, material)
            return

        print("Unknown argument.")
        print_help()


if __name__ == "__main__":
    main()
