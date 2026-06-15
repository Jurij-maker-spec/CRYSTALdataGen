from util.ref_db import backfill_hyperparameters_from_run_ids

# ["SiO2", "SiO2_PBE", "AlN", "AlN_PBE", "Al2O3", "Al2O3_PBE", "TiO2_rutil", "TiO2_rutil_PBE"]


for struct in ["SiO2_PBE"]:
    backfill_hyperparameters_from_run_ids(
        "data/ref_db.h5",
        structure=struct,
    )