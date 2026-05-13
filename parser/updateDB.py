from pathlib import Path
import sys
PYTHON_SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_SCRIPTS_ROOT))
from TOOLs.CRYSTALtoDB import *


if __name__ == "__main__":
    """
        NOT REFACTORED YET TO DO WHAT IT SHOULD DO
    """

    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide structure folder, e.g.\n"
            "python updateDB.py 00_structures"
        )

    struct_dir = Path(sys.argv[1]).resolve()

    if not struct_dir.exists():
        raise FileNotFoundError(f"{struct_dir} does not exist")

    db_root = Path('/home/jha/jha/python_scripts/CRYSTALreference')
    db_path = db_root / "CRYSTALreference.h5"

    print(f"structures folder: {struct_dir}")
    print(f"database:          {db_path}")

    structures = sorted(
        p for p in struct_dir.iterdir()
        if p.is_dir()
    )

    for structure_dir in structures:

        structure = structure_dir.name
        freq_dir = structure_dir / "freq"

        print(f"\n{structure}")

        if not freq_dir.exists():
            print("  Skipping: no freq directory")
            continue

        try:

            update_database(
                input_structure = structure,
                freq_dir = freq_dir,
                db_path = db_path,
                method = "CRYSTAL",
                mode_matching = False,
            )

            print("  Added to database")

        except FileNotFoundError as e:
            print(f"  Skipping: {e}")

        except Exception as e:
            print(f"  Failed: {e}")
            