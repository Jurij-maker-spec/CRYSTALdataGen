#!/usr/bin/env python3
"""
eval_model.py

Standalone evaluation script for existing MACELES sweep directories.

Examples:

python eval_model.py \
  --config AlN_10_90.yaml \
  --sweep-dir AlN_10_90/AlN_master_eval_full_260504_120000

python eval_model.py \
  --config AlN_10_90.yaml \
  --sweep-dir AlN_10_90/AlN_master_eval_full_260504_120000 \
  --dry-run

python eval_model.py \
  --config AlN_10_90.yaml \
  --sweep-dir AlN_10_90/AlN_master_eval_full_260504_120000 \
  --overwrite-summary \
  --no-skip-completed
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "training"
INFERENCE_DIR = PROJECT_ROOT / "inference"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from inference import evaluate_model
from util.eval import eval_cif_structure_name, run_eval_only_existing_sweep
from util.io import load_yaml
from util.common_helpers import (
    require_path,
    resolve_path
)


REQUIRED_ENV = "mace_env_3_12"
USE_DEPLOY_MODEL_IF_AVAILABLE = False



def resolve_config_path(config: Path) -> Path:
    if config.is_absolute():
        return config

    direct = PROJECT_ROOT / config
    if direct.exists():
        return direct

    nested = PROJECT_ROOT / "configs" / "master_cfg" / config
    if nested.exists():
        return nested

    raise FileNotFoundError(f"Could not find config file: {config}")


def resolve_existing_sweep_dir(sweep_dir: Path) -> Path:
    if sweep_dir.is_absolute():
        resolved = sweep_dir
    else:
        resolved = PROJECT_ROOT / "results" / sweep_dir

    if not resolved.exists():
        raise FileNotFoundError(f"Missing sweep directory: {resolved}")

    if not resolved.is_dir():
        raise NotADirectoryError(f"Sweep path is not a directory: {resolved}")

    return resolved


def check_conda_env() -> None:
    current_env = os.environ.get("CONDA_DEFAULT_ENV")

    if current_env != REQUIRED_ENV:
        sys.stderr.write(
            f"Error: This script must be run in conda env '{REQUIRED_ENV}', "
            f"but current env is '{current_env}'.\n"
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing MACELES sweep directory."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file.",
    )

    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Existing sweep master directory. Relative paths are resolved under PROJECT_ROOT/results.",
    )

    parser.add_argument(
        "--max-eval-runs",
        type=int,
        default=None,
        help="Optional limit for number of run directories evaluated.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered run dirs, configs, and model paths. No evaluation is run.",
    )

    parser.add_argument(
        "--overwrite-summary",
        action="store_true",
        help="Overwrite master_summary.json/master_summary.csv instead of appending/updating.",
    )

    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Evaluate runs even if eval summary and arrays already exist.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_conda_env()

    config_path = resolve_config_path(args.config)
    cfg: dict[str, Any] = load_yaml(config_path)

    required_keys = ["structure", "crystal_db_path", "eval_settings"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    structure = cfg["structure"]
    eval_structure = eval_cif_structure_name(structure)

    cif_path = require_path(
        cfg.get("cif_path", f"inference/CIFs/{eval_structure}.cif"),
        base=PROJECT_ROOT,
    )

    crystal_db_path = require_path(
        cfg["crystal_db_path"],
        base=PROJECT_ROOT,
    )

    eval_settings = deepcopy(cfg["eval_settings"])
    dry_run = args.dry_run or cfg.get("dry_run", False)

    sweep_root = resolve_existing_sweep_dir(args.sweep_dir)

    run_eval_only_existing_sweep(
        evaluate_model_func=evaluate_model,
        project_root=PROJECT_ROOT,
        sweep_root=sweep_root,
        structure=structure,
        cif_path=cif_path,
        crystal_db_path=crystal_db_path,
        eval_settings=eval_settings,
        max_eval_runs=args.max_eval_runs,
        dry_run=dry_run,
        skip_completed=not args.no_skip_completed,
        overwrite_summary=args.overwrite_summary,
        use_deploy_model_if_available=USE_DEPLOY_MODEL_IF_AVAILABLE,
    )


if __name__ == "__main__":
    main()
