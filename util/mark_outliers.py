#!/usr/bin/env python3

from pathlib import Path
import h5py

# ============================================================
# USER SETTINGS
# ============================================================
H5_PATH = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/data/dataset.h5")
STRUCTURE_NAME = "TiO2_rutil"

FAILED_DISTORTIONS = [
    "000081",
    "000130",
    "010088",
    "020021",
    "020125",
]

FAILED_STATUS = "failed"
FAILED_REASON = "manually flagged: near-zero displacement but elevated dE"

# If True, creates a backup copy next to the original file before editing
MAKE_BACKUP = False

# ============================================================
# MAIN
# ============================================================

def mark_outliers():
    if MAKE_BACKUP:
        backup_path = H5_PATH.with_suffix(H5_PATH.suffix + ".bak")
        if not backup_path.exists():
            print(f"Creating backup: {backup_path}")
            backup_path.write_bytes(H5_PATH.read_bytes())
        else:
            print(f"Backup already exists: {backup_path}")
    print('\n', STRUCTURE_NAME)
    with h5py.File(H5_PATH, "r+") as h5:
        distortions_root = h5["structures"][STRUCTURE_NAME]["distortions"]
        
        for distortion_id in FAILED_DISTORTIONS:
            if distortion_id not in distortions_root:
                print(f"[WARNING] Distortion {distortion_id} not found, skipping.")
                continue

            g = distortions_root[distortion_id]

            # ------------------------------------------------
            # status
            # ------------------------------------------------
            if "status" in g:
                del g["status"]
            g.create_dataset("status", data=FAILED_STATUS.encode("utf-8"))

            # ------------------------------------------------
            # include_in_training
            # ------------------------------------------------
            if "include_in_training" in g:
                del g["include_in_training"]
            g.create_dataset("include_in_training", data=0)

            # ------------------------------------------------
            # optional failure reason
            # ------------------------------------------------
            if "failure_reason" in g:
                del g["failure_reason"]
            g.create_dataset("failure_reason", data=FAILED_REASON.encode("utf-8"))

            print(f"[UPDATED] {distortion_id} -> status=failed, include_in_training=0")

    print("\nDone.")


if __name__ == "__main__":
    mark_outliers()
    