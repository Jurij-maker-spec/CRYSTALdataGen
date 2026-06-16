#!/usr/bin/env python3

from pathlib import Path
import h5py

ref_db = Path("data/ref_db.h5")

structure = "SiO2"
dataset_split = "SiO2_10_90"
sweep_id = "SiO2_master_eval_full_260427_170113"
run_id = "MACELES_bs2_ep173_ew3_fw100_rmax4_seed2_SiO2"

def looks_like_run_group(name: str) -> bool:
    parts = name.split("/")

    return (
        structure in parts
        and dataset_split in parts
        and sweep_id in parts
        and run_id in parts
    )


def print_group_contents(h5: h5py.File, group_path: str) -> None:
    print(f"\nRUN GROUP: {group_path}")

    obj = h5[group_path]

    def print_child(name, child):
        if isinstance(child, h5py.Dataset):
            print(f"  {name}: shape={child.shape}, dtype={child.dtype}")
        elif isinstance(child, h5py.Group):
            print(f"  {name}/")

    obj.visititems(print_child)


with h5py.File(ref_db, "r") as h5:
    matches = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and looks_like_run_group(name):
            matches.append(name)

    h5.visititems(visitor)

    # Keep only the shortest matching paths.
    # This avoids printing nested subgroups as separate run matches.
    if matches:
        min_depth = min(len(m.split("/")) for m in matches)
        matches = [m for m in matches if len(m.split("/")) == min_depth]

    print(f"Found {len(matches)} run-level matching groups.")

    for match in matches:
        print_group_contents(h5, match)

