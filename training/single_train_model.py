#!/usr/bin/env python3
"""
Train a periodic MACE / MACELES model on the CRYSTAL dataset.


Standalon script


"""
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

# ==============================
# USER CONFIGURATION
# ==============================

PROJECT_ROOT = Path("/home/jha/jha/python_scripts/CRYSTALdataGen")


train_file = "train_TiO2_rutil.xyz"
valid_file = "valid_interp_TiO2_rutil.xyz"

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / train_file
VALID_FILE = DATA_DIR / valid_file
# TEST_FILE = DATA_DIR / "test.xyz"   # optional later

# ---- model / chemistry ----
MODEL_TYPE = "MACELES"   # or "MACE"

# Elements in your current dataset:
# C=6, N=7, O=8, Na=11, Al=13, Si=14, P=15, Ti=22, Cu=29
# ATOMIC_NUMBERS = "[6,7,8,11,13,14,15,22,29]"
# ATOMIC_NUMBERS = "[7, 8, 13, 14, 22]"
ATOMIC_NUMBERS = "[8, 22]"
CHEM = "TiO2"
# Optional foundation model
FOUNDATION_MODEL = None   # e.g. "small", "medium", "large", "mh", ...

# You said you will insert these manually
# Example format:
# E0S = str({6: ..., 7: ..., 8: ..., 11: ..., 13: ..., 14: ..., 15: ..., 22: ..., 29: ...})
E0S = 'average'
# E0S = str({13:  -6579.795888626081,
#            14:  -7859.359462209514,
#             7: -1479.8286259990514,
#             8: -2034.7167916666829,
#            22: -23082.242849336617})

# ---- training ----
BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
MAX_EPOCHS = 123

ENERGY_WEIGHT = 1.0
FORCES_WEIGHT = 100.0

RMAX = 7.5              # new from 10.04 on
SEED = 2

# Set this True once you want to train on stresses too
USE_STRESS = False
STRESS_WEIGHT = 5.0

DEFAULT_DTYPE = "float64"
DEVICE = "cuda"
NUM_WORKERS = 0

USE_EMA = True
USE_SWA = False
RESTART_LATEST = False

# Key names in your extxyz export
ENERGY_KEY = "energy"
FORCES_KEY = "forces"
STRESS_KEY = "stress"

# ==============================
# EXPERIMENT NAME
# ==============================

timestamp = datetime.now().strftime("%y%m%d_%H%M")

run_bits = [
    MODEL_TYPE,
    f"bs{BATCH_SIZE}",
    f"ep{MAX_EPOCHS}",
    f"ew{int(ENERGY_WEIGHT)}",
    f"fw{int(FORCES_WEIGHT)}",
    f"{CHEM}"
]

if USE_STRESS:
    run_bits.append(f"sw{STRESS_WEIGHT}")

run_bits.append(timestamp)

RUN_NAME = "_".join(run_bits)
NAME = RUN_NAME

RESULT_DIR = PROJECT_ROOT / "results" / RUN_NAME
MODEL_DIR = RESULT_DIR / "models"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

DEPLOY_DIR = PROJECT_ROOT / "models"
DEPLOY_PATH = DEPLOY_DIR / f"{RUN_NAME}_singlehead.model"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nStarting experiment: {RUN_NAME}")
print(f"Results directory: {RESULT_DIR}\n")

# ==============================
# OPTIONAL ENVIRONMENT TUNING
# ==============================

# Uncomment if needed
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"

# ==============================
# INPUT CHECKS
# ==============================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Missing training file: {TRAIN_FILE}")

if not VALID_FILE.exists():
    raise FileNotFoundError(f"Missing validation file: {VALID_FILE}")

# ==============================
# BUILD TRAIN COMMAND
# ==============================
# "--r_max", str(RMAX),

train_cmd = [
    "python", "-m", "mace.cli.run_train",
    "--name", NAME,
    "--model", MODEL_TYPE,
    "--train_file", str(TRAIN_FILE),
    "--valid_file", str(VALID_FILE),
    "--atomic_numbers", ATOMIC_NUMBERS,
    "--max_num_epochs", str(MAX_EPOCHS),
    "--energy_weight", str(ENERGY_WEIGHT),
    "--forces_weight", str(FORCES_WEIGHT),
    "--energy_key", ENERGY_KEY,
    "--forces_key", FORCES_KEY,
    "--E0s", E0S,
    "--seed", str(SEED),
    "--r_max", str(RMAX), 
    "--device", DEVICE,
    "--batch_size", str(BATCH_SIZE),
    "--valid_batch_size", str(VALID_BATCH_SIZE),
    "--default_dtype", DEFAULT_DTYPE,
    "--num_workers", str(NUM_WORKERS),
    "--work_dir", str(RESULT_DIR),
    "--log_dir", str(RESULT_DIR),
    "--model_dir", str(MODEL_DIR),
    "--checkpoints_dir", str(CHECKPOINT_DIR),
    "--results_dir", str(RESULT_DIR),
]

if FOUNDATION_MODEL is not None:
    train_cmd += ["--foundation_model", FOUNDATION_MODEL]

if USE_STRESS:
    train_cmd += [
        "--stress_key", STRESS_KEY,
        "--stress_weight", str(STRESS_WEIGHT),
    ]
else:
    train_cmd += [
        "--stress_weight", "0.0",
    ]

if RESTART_LATEST:
    train_cmd.append("--restart_latest")

if USE_EMA:
    train_cmd.append("--ema")

if USE_SWA:
    train_cmd.append("--swa")

# If you later add an external test file:
# train_cmd += ["--test_file", str(TEST_FILE)]

# ==============================
# LOG COMMAND
# ==============================

train_log_path = RESULT_DIR / "train.log"
cmd_txt_path = RESULT_DIR / "train_command.txt"

with open(cmd_txt_path, "w") as f:
    f.write(" ".join(shlex.quote(x) for x in train_cmd) + "\n")

print("Training command:")
print(" ".join(shlex.quote(x) for x in train_cmd))
print()

# ==============================
# RUN TRAINING
# ==============================

with open(train_log_path, "w") as log_file:
    subprocess.run(
        train_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        check=True,
        cwd=str(RESULT_DIR),
    )

print("Training finished successfully.")

# ==============================
# FIND TRAINED MODEL
# ==============================

candidate_main = RESULT_DIR / f"{RUN_NAME}.model"
candidate_model_dir = MODEL_DIR / f"{RUN_NAME}.model"

if candidate_main.exists():
    trained_model_path = candidate_main
elif candidate_model_dir.exists():
    trained_model_path = candidate_model_dir
else:
    model_candidates = sorted(MODEL_DIR.glob("*.model"), key=lambda p: p.stat().st_mtime)
    if not model_candidates:
        raise FileNotFoundError(f"No .model file found in {MODEL_DIR}")
    trained_model_path = model_candidates[-1]

print(f"Using trained model for head extraction:\n{trained_model_path}\n")

# ==============================
# EXTRACT SINGLE HEAD
# ==============================

extract_cmd = [
    "python", "-m", "mace.cli.mace_select_head",
    "--model", str(trained_model_path),
    "--head", "default",
    "--output", str(DEPLOY_PATH),
]

extract_log_path = RESULT_DIR / "extract.log"

with open(extract_log_path, "w") as log_file:
    subprocess.run(
        extract_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        check=True,
        cwd=str(RESULT_DIR),
    )

print("Single-head model extracted.")
print(f"Deployment model saved to:\n{DEPLOY_PATH}\n")
