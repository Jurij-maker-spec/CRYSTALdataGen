from pathlib import Path
import sys
PYTHON_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PYTHON_SCRIPTS_ROOT
sys.path.insert(0, str(PYTHON_SCRIPTS_ROOT))
from util.ref_db import (
    backfill_hyperparameters_from_run_ids,
    merge_backfill_hyperparameters_from_run_ids,
)

structures = [
    "SiO2",
    "SiO2_PBE",
    "AlN",
    "AlN_PBE",
    "Al2O3",
    "Al2O3_PBE",
    "TiO2_rutil",
    "TiO2_rutil_PBE",
]

for struct in structures:
    # backfill_hyperparameters_from_run_ids(
    #     "data/ref_db.h5",
    #     structure=struct,
    #     train_size_map_path="data/train_size_map.txt",
    #     overwrite=True,
    #     dry_run=False,
    # )
    merge_backfill_hyperparameters_from_run_ids(
        PROJECT_ROOT / "data/ref_db.h5",
        structure=struct,
        train_size_map_path= PROJECT_ROOT / "data/train_size_map.txt",
        dry_run=False,
    )
    