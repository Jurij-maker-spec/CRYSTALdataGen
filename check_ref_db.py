import h5py
from pathlib import Path

ref_db = Path("data/ref_db.h5")
structure = "SiO2"

with h5py.File(ref_db, "r") as h5:
    root = h5[f"structures/{structure}/evaluations"]

    for split_name, split_group in root.items():
        for sweep_name, sweep_group in split_group.items():
            for run_name, run_group in sweep_group.items():
                print("\nRUN:", run_name)
                print("ATTRS:")
                for k, v in run_group.attrs.items():
                    print(" ", k, "=", v)

                if "hyperparameters" in run_group:
                    print("hypeparams:")
                    for k, v in run_group["hyperparameters"].attrs.items():
                        print(" ", k, "=", v)

                raise SystemExit