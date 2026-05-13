 # CRYSTAL DFT Dataset Tools

Utilities for generating distorted crystal structures, building HDF5 datasets,
training/evaluating MACELES models, and comparing predicted IR/phonon properties
against CRYSTAL reference calculations.

## Main workflows

- Generate distorted structures from YAML configs
- Build training HDF5 databases
- Build CRYSTAL reference databases
- Export train/valid XYZ files
- Run MACELES training sweeps
- Evaluate IR spectra, phonons, BECs, and mode matching

## Environment

See `environment.yml`.

## Typical usage

```bash
python generate_distortions.py configs/example.yaml
python build_train_db.py configs/build_db.yaml
python run_master_eval.py configs/sweep.yaml
