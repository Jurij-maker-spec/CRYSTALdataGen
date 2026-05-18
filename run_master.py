#!/usr/bin/env python3
"""
run_master.py

Top-level orchestration script for:
1. preparing a hyperparameter sweep from config
2. optionally dry-running the sweep setup
3. optionally running a small smoke test
4. running training jobs in parallel
5. starting evaluation immediately when each training job finishes
6. saving a sweep-wide summary
7. explicitly naming the best model after the sweep
"""

from __future__ import annotations

import re
import csv
import numpy as np
import json
import math
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import yaml
import os

REQUIRED_ENV = "mace_env_3_12"
# ============================================================
# PATH SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "training"
INFERENCE_DIR = PROJECT_ROOT / "inference"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from sweep_training import prepare_sweep, run_sweep_parallel, BASE_CONFIG, DEFAULT_SWEEP_GRID
from inference import evaluate_model


# ============================================================
# USER CONFIG
# ============================================================
SMOKE_MAX_RUNS = 3
SMOKE_MAX_PARALLEL_TRAININGS = 1
USE_DEPLOY_MODEL_IF_AVAILABLE = False


# ============================================================
# YAML CONFIG
# ============================================================

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")

    return cfg


def resolve_path(value: str | None, base: Path = PROJECT_ROOT) -> Path | None:
    if value is None:
        return None

    p = Path(value)
    if p.is_absolute():
        return p

    return base / p



# ============================================================
# HELPERS
# ============================================================

def require_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    p = resolve_path(value, base=base)
    if p is None:
        raise ValueError("Required path is None")
    return p


def parse_mace_rmse_table_from_train_log(train_log_path: Path) -> dict[str, Any]:
    """
    Parse final MACE RMSE table from train.log.

    Expected rows:
    | train_Default | 1.2 | 4.0 | 0.27 |
    | valid_Default | 1.4 | 4.4 | 0.30 |
    """
    metrics = {
        "train_rmse_e_mev_atom": None,
        "train_rmse_f_mev_A": None,
        "train_relative_f_rmse_percent": None,
        "valid_rmse_e_mev_atom": None,
        "valid_rmse_f_mev_A": None,
        "valid_relative_f_rmse_percent": None,
    }

    if not train_log_path.exists():
        return metrics

    text = train_log_path.read_text(encoding="utf-8", errors="replace")

    row_pattern = re.compile(
        r"\|\s*(train_Default|valid_Default)\s*"
        r"\|\s*([+-]?\d+(?:\.\d+)?)\s*"
        r"\|\s*([+-]?\d+(?:\.\d+)?)\s*"
        r"\|\s*([+-]?\d+(?:\.\d+)?)\s*\|"
    )

    matches = row_pattern.findall(text)

    # If several tables exist, keep the last occurrence per config_type.
    for config_type, rmse_e, rmse_f, rel_f in matches:
        prefix = "train" if config_type == "train_Default" else "valid"

        metrics[f"{prefix}_rmse_e_mev_atom"] = float(rmse_e)
        metrics[f"{prefix}_rmse_f_mev_A"] = float(rmse_f)
        metrics[f"{prefix}_relative_f_rmse_percent"] = float(rel_f)

    return metrics


def get_train_log_metrics(train_result: dict[str, Any]) -> dict[str, Any]:
    result_dir = train_result.get("result_dir")
    if not result_dir:
        return parse_mace_rmse_table_from_train_log(Path("__missing_train_log__"))

    return parse_mace_rmse_table_from_train_log(Path(result_dir) / "train.log")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]

    return obj


def save_json(payload: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_master_csv(row: dict[str, Any], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)

    fieldnames = [
        "run_name",
        "train_status",
        "eval_status",
        "model_path",
        "result_dir",
        "seed",
        "r_max",
        "batch_size",
        "valid_batch_size",
        "energy_weight",
        "forces_weight",
        "use_stress",
        "stress_weight",
        "max_epochs",
        "train_rmse_e_mev_atom",
        "train_rmse_f_mev_A",
        "train_relative_f_rmse_percent",
        "valid_rmse_e_mev_atom",
        "valid_rmse_f_mev_A",
        "valid_relative_f_rmse_percent",
        "n_imag_modes",
        "n_ir_active_modes",
        "zpe_eV",
        "spectrum_rel_l2",
        "freq_mae_ir_cm1",
        "freq_rmse_ir_cm1",
        "freq_mae_ir_weighted_cm1",
        "intensity_pearson_r",
        "intensity_spearman_r",
        "matched_mode_count",
        "composite_score",
        "summary_json",
        "ir_plot",
        "error",
    ]

    file_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_run_config_from_result(train_result: dict[str, Any]) -> dict[str, Any] | None:
    run_config_path = train_result.get("run_config_path")
    if not run_config_path:
        return None

    path = Path(run_config_path)
    if not path.exists():
        return None

    return load_json(path)


def choose_model_path(train_result: dict[str, Any]) -> Path:
    deploy_model_path = train_result.get("deploy_model_path")
    trained_model_path = train_result.get("trained_model_path")

    if USE_DEPLOY_MODEL_IF_AVAILABLE and deploy_model_path:
        p = Path(deploy_model_path)
        if p.exists():
            return p

    if trained_model_path:
        p = Path(trained_model_path)
        if p.exists():
            return p

    if deploy_model_path:
        p = Path(deploy_model_path)
        if p.exists():
            return p

    raise FileNotFoundError("No usable model path found in train_result.")


def extract_run_hparams(run_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if run_cfg is None:
        return {
            "seed": None,
            "r_max": None,
            "batch_size": None,
            "valid_batch_size": None,
            "energy_weight": None,
            "forces_weight": None,
            "use_stress": None,
            "stress_weight": None,
            "max_epochs": None,
        }

    return {
        "seed": run_cfg.get("seed"),
        "r_max": run_cfg.get("r_max"),
        "batch_size": run_cfg.get("batch_size"),
        "valid_batch_size": run_cfg.get("valid_batch_size"),
        "energy_weight": run_cfg.get("energy_weight"),
        "forces_weight": run_cfg.get("forces_weight"),
        "use_stress": run_cfg.get("use_stress"),
        "stress_weight": run_cfg.get("stress_weight"),
        "max_epochs": run_cfg.get("max_epochs"),
    }


def safe_get_ranking_metrics(eval_result: dict[str, Any] | None) -> dict[str, Any]:
    if eval_result is None:
        return {
            "freq_mae_ir_cm1": None,
            "freq_rmse_ir_cm1": None,
            "freq_mae_ir_weighted_cm1": None,
            "intensity_pearson_r": None,
            "intensity_spearman_r": None,
            "matched_mode_count": None,
            "composite_score": None,
        }

    rm = eval_result.get("ranking_metrics", {})
    return {
        "freq_mae_ir_cm1": rm.get("freq_mae_ir_cm1"),
        "freq_rmse_ir_cm1": rm.get("freq_rmse_ir_cm1"),
        "freq_mae_ir_weighted_cm1": rm.get("freq_mae_ir_weighted_cm1"),
        "intensity_pearson_r": rm.get("intensity_pearson_r"),
        "intensity_spearman_r": rm.get("intensity_spearman_r"),
        "matched_mode_count": rm.get("matched_mode_count"),
        "composite_score": rm.get("composite_score"),
    }


def build_eval_row(
    train_result: dict[str, Any],
    eval_result: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    run_cfg = load_run_config_from_result(train_result)
    hp = extract_run_hparams(run_cfg)
    rm = safe_get_ranking_metrics(eval_result)
    train_log_metrics = get_train_log_metrics(train_result)

    row = {
        "run_name": train_result.get("run_name"),
        "train_status": train_result.get("status"),
        "eval_status": "not_run" if eval_result is None and error is None else ("failed" if error else "ok"),
        "model_path": None,
        "result_dir": train_result.get("result_dir"),
        "seed": hp["seed"],
        "r_max": hp["r_max"],
        "batch_size": hp["batch_size"],
        "valid_batch_size": hp["valid_batch_size"],
        "energy_weight": hp["energy_weight"],
        "forces_weight": hp["forces_weight"],
        "use_stress": hp["use_stress"],
        "stress_weight": hp["stress_weight"],
        "max_epochs": hp["max_epochs"],
        "train_rmse_e_mev_atom": train_log_metrics["train_rmse_e_mev_atom"],
        "train_rmse_f_mev_A": train_log_metrics["train_rmse_f_mev_A"],
        "train_relative_f_rmse_percent": train_log_metrics["train_relative_f_rmse_percent"],
        "valid_rmse_e_mev_atom": train_log_metrics["valid_rmse_e_mev_atom"],
        "valid_rmse_f_mev_A": train_log_metrics["valid_rmse_f_mev_A"],
        "valid_relative_f_rmse_percent": train_log_metrics["valid_relative_f_rmse_percent"],
        "n_imag_modes": None,
        "n_ir_active_modes": None,
        "zpe_eV": None,
        "spectrum_rel_l2": None,
        "freq_mae_ir_cm1": rm["freq_mae_ir_cm1"],
        "freq_rmse_ir_cm1": rm["freq_rmse_ir_cm1"],
        "freq_mae_ir_weighted_cm1": rm["freq_mae_ir_weighted_cm1"],
        "intensity_pearson_r": rm["intensity_pearson_r"],
        "intensity_spearman_r": rm["intensity_spearman_r"],
        "matched_mode_count": rm["matched_mode_count"],
        "composite_score": rm["composite_score"],
        "summary_json": None,
        "ir_plot": None,
        "error": error,
    }

    if train_result.get("status") == "ok":
        try:
            row["model_path"] = str(choose_model_path(train_result))
        except Exception as exc:
            row["error"] = repr(exc)

    if eval_result is not None:
        row["n_imag_modes"] = eval_result.get("n_imag_modes")
        row["n_ir_active_modes"] = eval_result.get("n_ir_active_modes")
        row["zpe_eV"] = eval_result.get("zpe_eV")

        crystal_cmp = eval_result.get("crystal_comparison", {})
        row["spectrum_rel_l2"] = crystal_cmp.get("spectrum_rel_l2")

        artifacts = eval_result.get("artifacts", {})
        row["summary_json"] = artifacts.get("summary_json")
        row["ir_plot"] = artifacts.get("ir_plot")

    return row


def apply_smoke_overrides(
    base_config: dict[str, Any],
    sweep_grid: dict[str, list[Any]],
) -> tuple[dict[str, Any], dict[str, list[Any]], int, int | None]:
    base = deepcopy(base_config)
    grid = deepcopy(sweep_grid)

    base["max_epochs"] = 2
    base["run_extraction"] = True

    grid["seed"] = [1]
    grid["r_max"] = [5.0, 6.0, 7.0]
    grid["energy_weight"] = [1.0]
    grid["forces_weight"] = [100.0]
    grid["batch_size"] = [2]
    grid["valid_batch_size"] = [2]
    grid["use_stress"] = [False]
    grid["stress_weight"] = [2.0]

    return base, grid, SMOKE_MAX_PARALLEL_TRAININGS, SMOKE_MAX_RUNS


def preview_run_commands(run_config_paths: list[Path]) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []

    for cfg_path in run_config_paths:
        cfg = load_json(cfg_path)

        data_dir = Path(cfg["data_dir"]).resolve()
        train_file = data_dir / cfg["train_file"]
        valid_file = data_dir / cfg["valid_file"]
        result_dir = Path(cfg["run_dir"]).resolve()
        model_dir = result_dir / "models"
        checkpoint_dir = result_dir / "checkpoints"

        train_cmd = [
            sys.executable, "-m", "mace.cli.run_train",
            "--name", cfg["run_name"],
            "--model", cfg["model_type"],
            "--train_file", str(train_file),
            "--valid_file", str(valid_file),
            "--atomic_numbers", json.dumps(cfg["atomic_numbers"]),
            "--max_num_epochs", str(cfg["max_epochs"]),
            "--energy_weight", str(cfg["energy_weight"]),
            "--forces_weight", str(cfg["forces_weight"]),
            "--energy_key", cfg["energy_key"],
            "--forces_key", cfg["forces_key"],
            "--E0s", cfg["E0s"],
            "--device", cfg["device"],
            "--batch_size", str(cfg["batch_size"]),
            "--valid_batch_size", str(cfg["valid_batch_size"]),
            "--default_dtype", cfg["default_dtype"],
            "--num_workers", str(cfg["num_workers"]),
            "--work_dir", str(result_dir),
            "--log_dir", str(result_dir),
            "--model_dir", str(model_dir),
            "--checkpoints_dir", str(checkpoint_dir),
            "--results_dir", str(result_dir),
            "--seed", str(cfg["seed"]),
            "--r_max", str(cfg["r_max"]),
        ]

        if cfg.get("foundation_model") is not None:
            train_cmd += ["--foundation_model", str(cfg["foundation_model"])]

        if cfg.get("use_stress", False):
            train_cmd += [
                "--stress_key", cfg["stress_key"],
                "--stress_weight", str(cfg["stress_weight"]),
            ]
        else:
            train_cmd += ["--stress_weight", "0.0"]

        if cfg.get("restart_latest", False):
            train_cmd.append("--restart_latest")
        if cfg.get("use_ema", False):
            train_cmd.append("--ema")
        if cfg.get("use_swa", False):
            train_cmd.append("--swa")

        previews.append({
            "run_name": cfg["run_name"],
            "run_config_path": str(cfg_path),
            "result_dir": str(result_dir),
            "train_file_exists": train_file.exists(),
            "valid_file_exists": valid_file.exists(),
            "train_command": train_cmd,
        })

    return previews


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def build_best_model_record(eval_record: dict[str, Any]) -> dict[str, Any]:
    train_result = eval_record["train_result"]
    eval_result = eval_record["eval_result"]
    row = build_eval_row(train_result, eval_result, error=None)

    return {
        "run_name": eval_record.get("run_name"),
        "evaluated_at": eval_record.get("evaluated_at"),
        "model_path": row.get("model_path"),
        "result_dir": row.get("result_dir"),
        "summary_json": row.get("summary_json"),
        "ir_plot": row.get("ir_plot"),
        "metrics": {
            "n_imag_modes": row.get("n_imag_modes"),
            "n_ir_active_modes": row.get("n_ir_active_modes"),
            "zpe_eV": row.get("zpe_eV"),
            "spectrum_rel_l2": row.get("spectrum_rel_l2"),
            "freq_mae_ir_cm1": row.get("freq_mae_ir_cm1"),
            "freq_rmse_ir_cm1": row.get("freq_rmse_ir_cm1"),
            "freq_mae_ir_weighted_cm1": row.get("freq_mae_ir_weighted_cm1"),
            "intensity_pearson_r": row.get("intensity_pearson_r"),
            "intensity_spearman_r": row.get("intensity_spearman_r"),
            "matched_mode_count": row.get("matched_mode_count"),
            "composite_score": row.get("composite_score"),
        },
    }


def evaluation_sort_key(eval_record: dict[str, Any]) -> tuple:
    """
    Lower is better.

    Primary:
      - composite_score if available

    Fallbacks:
      - freq_mae_ir_cm1
      - spectrum_rel_l2
      - n_imag_modes

    Imaginary modes are a penalty, not a rejection.
    """
    eval_result = eval_record["eval_result"]
    row = build_eval_row(eval_record["train_result"], eval_result, error=None)

    composite_score = row.get("composite_score")
    freq_mae = row.get("freq_mae_ir_cm1")
    spectrum_l2 = row.get("spectrum_rel_l2")
    n_imag = row.get("n_imag_modes")
    matched_mode_count = row.get("matched_mode_count")

    has_composite = 0 if is_finite_number(composite_score) else 1
    composite_val = float(composite_score) if is_finite_number(composite_score) else float("inf")

    has_freq_mae = 0 if is_finite_number(freq_mae) else 1
    freq_mae_val = float(freq_mae) if is_finite_number(freq_mae) else float("inf")

    has_spectrum = 0 if is_finite_number(spectrum_l2) else 1
    spectrum_val = float(spectrum_l2) if is_finite_number(spectrum_l2) else float("inf")

    imag_val = int(n_imag) if isinstance(n_imag, int) else 10**9
    matched_val = -int(matched_mode_count) if isinstance(matched_mode_count, int) else 10**9

    return (
        has_composite,
        composite_val,
        has_freq_mae,
        freq_mae_val,
        has_spectrum,
        spectrum_val,
        imag_val,
        matched_val,
        row.get("run_name", ""),
    )


def choose_best_model(master_summary: dict[str, Any]) -> dict[str, Any] | None:
    ok_records = [
        rec for rec in master_summary.get("evaluation_results", [])
        if rec.get("status") == "ok" and "eval_result" in rec
    ]

    if not ok_records:
        return None

    best = min(ok_records, key=evaluation_sort_key)
    return build_best_model_record(best)


def eval_cif_structure_name(structure: str) -> str:
    """
    Evaluation CIFs use the original structure name, not functional suffixes.

    Example:
        SiO2_PBE -> SiO2
        AlN_PBE  -> AlN
    """
    for suffix in ("_PBE", "_PBESOLXC", "_PBESOL", "_HSESOL", "_HSE"):
        if structure.upper().endswith(suffix):
            return structure[: -len(suffix)]
    return structure


# ============================================================
# EVAL ONLY HELPERS
# ============================================================

def find_existing_run_dirs(sweep_root: Path) -> list[Path]:
    run_dirs: list[Path] = []

    for child in sorted(sweep_root.iterdir()):
        if not child.is_dir():
            continue

        if not (child / "run_config.json").exists():
            continue

        has_model = False

        models_dir = child / "models"
        checkpoints_dir = child / "checkpoints"

        if models_dir.exists() and any(models_dir.glob("*.model")):
            has_model = True

        #if checkpoints_dir.exists() and any(checkpoints_dir.glob("*.model")):
        #    has_model = True

        if has_model:
            run_dirs.append(child)

    return run_dirs


def choose_existing_model_path(run_dir: Path) -> Path:
    models = sorted((run_dir / "models").glob("*.model"))
    if models:
        return models[-1]

    checkpoint_models = sorted((run_dir / "checkpoints").glob("*.model"))
    if checkpoint_models:
        return checkpoint_models[-1]

    raise FileNotFoundError(f"No .model file found in {run_dir}")


def build_train_result_from_existing_run(run_dir: Path) -> dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    run_cfg = load_json(run_config_path)

    model_path = choose_existing_model_path(run_dir)

    train_result_path = run_dir / "train_result.json"
    old_train_result = load_json(train_result_path) if train_result_path.exists() else {}

    return {
        "run_name": run_cfg.get("run_name", run_dir.name),
        "status": "ok",
        "result_dir": str(run_dir),
        "run_config_path": str(run_config_path),
        "trained_model_path": str(model_path),
        "deploy_model_path": old_train_result.get("deploy_model_path"),
        "error": None,
    }



# ============================================================
# EVAL ONLY
# ============================================================

def preview_existing_eval_runs(
        sweep_root: Path,
        structure: str,
        max_eval_runs: int | None = None,
        ) -> list[dict[str, Any]]:
    run_dirs = find_existing_run_dirs(sweep_root)

    if max_eval_runs is not None:
        run_dirs = run_dirs[:max_eval_runs]

    previews: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_config_path = run_dir / "run_config.json"

        try:
            run_cfg = load_json(run_config_path)
            model_path = choose_existing_model_path(run_dir)
            eval_summary_path = run_dir / f"{structure}_eval_summary.json"
            eval_arrays_path = run_dir / f"{structure}_eval_arrays.npz"

            previews.append({
                "run_name": run_cfg.get("run_name", run_dir.name),
                "run_dir": str(run_dir),
                "run_config": str(run_config_path),
                "model_path": str(model_path),
                "eval_summary_exists": eval_summary_path.exists(),
                "eval_arrays_exists": eval_arrays_path.exists(),
            })

        except Exception as exc:
            previews.append({
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "run_config": str(run_config_path),
                "model_path": None,
                "error": repr(exc),
            })

    return previews


def run_eval_only_existing_sweep(
    sweep_root: Path,
    structure: str,
    cif_path: Path,
    crystal_db_path: Path,
    eval_settings: dict[str, Any],
    max_eval_runs: int | None = None,
    dry_run: bool = False,
) -> None:
    
    if not cif_path.exists():
        raise FileNotFoundError(f"Missing CIF: {cif_path}")
    if not crystal_db_path.exists():
        raise FileNotFoundError(f"Missing CRYSTAL DB: {crystal_db_path}")

    run_dirs = find_existing_run_dirs(sweep_root)

    if dry_run:
        previews = preview_existing_eval_runs(
            sweep_root=sweep_root,
            structure=structure,
            max_eval_runs=max_eval_runs,

        )

        print(f"Eval-only dry run sweep root: {sweep_root}")
        print(f"Runs found: {len(previews)}")
        print()

        for i, preview in enumerate(previews, start=1):
            print(f"[{i}] {preview['run_name']}")
            print(f"  run_dir: {preview['run_dir']}")
            print(f"  run_config: {preview['run_config']}")
            print(f"  model_path: {preview.get('model_path')}")
            print(f"  eval_summary_exists: {preview.get('eval_summary_exists')}")
            print(f"  eval_arrays_exists: {preview.get('eval_arrays_exists')}")

            if preview.get("error"):
                print(f"  error: {preview['error']}")

            print()

        return


    if max_eval_runs is not None:
        run_dirs = run_dirs[:max_eval_runs]

    master_summary_path = sweep_root / "master_summary.json"
    master_csv_path = sweep_root / "master_summary.csv"
    best_model_path = sweep_root / "best_model.json"

    if master_csv_path.exists():
        master_csv_path.unlink()

    master_summary: dict[str, Any] = {
        "sweep_name": sweep_root.name,
        "created_at": now_iso(),
        "mode": "eval_only",
        "project_root": str(PROJECT_ROOT),
        "sweep_root": str(sweep_root),
        "structure": structure,
        "cif_path": str(cif_path),
        "crystal_db_path": str(crystal_db_path),
        "max_eval_runs": max_eval_runs,
        "use_deploy_model_if_available": USE_DEPLOY_MODEL_IF_AVAILABLE,
        "eval_settings": deepcopy(eval_settings),
        "n_existing_runs_found": len(run_dirs),
        "training_results": [],
        "evaluation_results": [],
        "best_model": None,
    }

    save_json(master_summary, master_summary_path)

    print(f"Eval-only sweep root: {sweep_root}")
    print(f"Run directories found: {len(run_dirs)}")

    for run_dir in run_dirs:
        train_result = build_train_result_from_existing_run(run_dir)
        master_summary["training_results"].append(train_result)
        save_json(master_summary, master_summary_path)

        run_name = train_result.get("run_name", run_dir.name)

        try:
            model_path = choose_model_path(train_result)
            result_dir = Path(train_result["result_dir"])

            print(f"\n=== Starting evaluation for existing run: {run_name} ===")
            print(f"Using model: {model_path}")

            eval_result = evaluate_model(
                model_path=model_path,
                structure=structure,
                cif_path=cif_path,
                output_dir=result_dir,
                crystal_db_path=crystal_db_path,
                device=eval_settings["device"],
                default_dtype=eval_settings["default_dtype"],
                frechet=eval_settings["frechet"],
                fmax=eval_settings["fmax"],
                calculator_mode=eval_settings["calculator_mode"],
                compare_crystal_modes=eval_settings["compare_crystal_modes"],
                crystal_hess_path=eval_settings["crystal_hess_path"],
                crystal_freq_out_path=eval_settings["crystal_freq_out_path"],
                crystal_hessian_units=eval_settings["crystal_hessian_units"],
                # reserved plugin interface
                run_phonopy=eval_settings.get("run_phonopy", False),
                phonopy_plugin=eval_settings.get("phonopy_plugin"),
                write_ref_db=eval_settings.get("write_ref_db", False),
                ref_db_path=eval_settings.get("ref_db_path"),
                run_id=run_name,
                dataset_split=sweep_root.parent.name,
                sweep_id=sweep_root.name,
            )

            eval_record = {
                "run_name": run_name,
                "status": "ok",
                "evaluated_at": now_iso(),
                "train_result": train_result,
                "eval_result": {
                    "n_imag_modes": eval_result.get("n_imag_modes"),
                    "n_ir_active_modes": eval_result.get("n_ir_active_modes"),
                    "zpe_eV": eval_result.get("zpe_eV"),
                    "crystal_comparison": {
                        "spectrum_rel_l2": eval_result.get("crystal_comparison", {}).get("spectrum_rel_l2"),
                    },
                    "ranking_metrics": eval_result.get("ranking_metrics", {}),
                    "artifacts": eval_result.get("artifacts", {}),
                },
                "ref_db": eval_result.get("ref_db", {}),
            }

            master_summary["evaluation_results"].append(eval_record)

            row = build_eval_row(
                train_result=train_result,
                eval_result=eval_result,
                error=None,
            )
            append_master_csv(row, master_csv_path)

            best_model = choose_best_model(master_summary)
            master_summary["best_model"] = best_model
            save_json(master_summary, master_summary_path)

            if best_model is not None:
                save_json(best_model, best_model_path)

            print(f"Evaluation finished for: {run_name}")

            if best_model is not None:
                print(f"Current best model: {best_model['run_name']}")
                print(f"Current best model path: {best_model['model_path']}")

        except Exception as exc:
            eval_record = {
                "run_name": run_name,
                "status": "failed",
                "evaluated_at": now_iso(),
                "train_result": train_result,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

            master_summary["evaluation_results"].append(eval_record)

            row = build_eval_row(
                train_result=train_result,
                eval_result=None,
                error=repr(exc),
            )
            append_master_csv(row, master_csv_path)

            best_model = choose_best_model(master_summary)
            master_summary["best_model"] = best_model
            save_json(master_summary, master_summary_path)

            if best_model is not None:
                save_json(best_model, best_model_path)

            print(f"Evaluation failed for: {run_name}")
            print(repr(exc))

    master_summary["best_model"] = choose_best_model(master_summary)
    master_summary["finished_at"] = now_iso()
    save_json(master_summary, master_summary_path)

    if master_summary["best_model"] is not None:
        save_json(master_summary["best_model"], best_model_path)

    print("\nEval-only sweep finished.")
    print(f"Master summary JSON: {master_summary_path}")
    print(f"Master summary CSV : {master_csv_path}")
    print(f"Best model JSON    : {best_model_path}")



# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a MACELES sweep or rerun evaluation for an existing sweep."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file for this master evaluation run.",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Rerun only the evaluation/inference part for one existing sweep master directory.",
    )

    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=None,
        help=(
            "Existing sweep master directory for --eval-only. "
            "Can be absolute or relative to PROJECT_ROOT/results."
        ),
    )

    parser.add_argument(
        "--max-eval-runs",
        type=int,
        default=None,
        help="Optional limit for number of run directories evaluated in --eval-only mode.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --eval-only, only print discovered run dirs, configs, and model paths. No evaluation is run.",
    )

    return parser.parse_args()


def resolve_results_group_root(group_dir: str | None) -> Path:
    results_root = PROJECT_ROOT / "results"

    if group_dir is None or str(group_dir).strip() == "":
        return results_root

    return results_root / group_dir


def resolve_existing_sweep_dir(sweep_dir: Path) -> Path:
    sweep_dir = Path(sweep_dir)

    if sweep_dir.is_absolute():
        resolved = sweep_dir
    else:
        resolved = PROJECT_ROOT / "results" / sweep_dir

    if not resolved.exists():
        raise FileNotFoundError(f"Missing sweep directory: {resolved}")

    if not resolved.is_dir():
        raise NotADirectoryError(f"Sweep path is not a directory: {resolved}")

    return resolved


def main() -> None:
    args = parse_args()
    config_root = PROJECT_ROOT / 'configs/master_cfg' / str(args.config)
    cfg = load_yaml(config_root)

    # ============================================================
    # Check correct environment
    # ============================================================
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env != REQUIRED_ENV:
        sys.stderr.write(
            f"Error: This script must be run in conda env '{REQUIRED_ENV}', "
            f"but current env is '{current_env}'.\n"
        )
        sys.exit(1)

    if os.environ.get("CONDA_DEFAULT_ENV") != REQUIRED_ENV:
        os.execvp(
            "conda",
            ["conda", "run", "-n", REQUIRED_ENV, "python"] + sys.argv
        )

    # ============================================================
    # Load from YAML
    # ============================================================
    required_keys = ["structure", "crystal_db_path", "eval_settings", "base_config", "sweep_grid"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    structure = cfg["structure"]
    results_group_dir = cfg.get("results_group_dir", structure)
    eval_cif_structure = eval_cif_structure_name(structure)
    cif_path = require_path(
        cfg.get("cif_path", f"inference/CIFs/{eval_cif_structure}.cif")
    )
    crystal_db_path = require_path(
        cfg["crystal_db_path"],
        base=PROJECT_ROOT,
    )
    dry_run = args.dry_run or cfg.get("dry_run", False)
    smoke_test = cfg.get("smoke_test", False)
    max_runs = cfg.get("max_runs")
    max_parallel_trainings = cfg.get("max_parallel_trainings", 1)
    eval_settings = cfg["eval_settings"]
    
    base_config = deepcopy(BASE_CONFIG)
    base_config.update(cfg["base_config"])
    base_config.update({
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(PROJECT_ROOT / "data"),
        "deploy_root": str(PROJECT_ROOT / "models"),
    })

    sweep_grid = deepcopy(DEFAULT_SWEEP_GRID)
    sweep_grid.update(cfg["sweep_grid"])

    
    if args.eval_only:
        if args.sweep_dir is None:
            raise ValueError("--sweep-dir is required with --eval-only")

        sweep_root = resolve_existing_sweep_dir(args.sweep_dir)
        run_eval_only_existing_sweep(
            sweep_root=sweep_root,
            structure=structure,
            cif_path=cif_path,
            crystal_db_path=crystal_db_path,
            eval_settings=eval_settings,
            max_eval_runs=args.max_eval_runs,
            dry_run=dry_run,
        )
        return

    if not cif_path.exists():
        raise FileNotFoundError(f"Missing CIF: {cif_path}")
    if not crystal_db_path.exists():
        raise FileNotFoundError(f"Missing CRYSTAL DB: {crystal_db_path}")

    group_root = resolve_results_group_root(results_group_dir)
    ensure_dir(group_root)
    base_config["results_root"] = str(group_root)

    sweep_grid = deepcopy(sweep_grid)
    max_parallel = max_parallel_trainings
    effective_max_runs = max_runs

    mode_tag = "full"

    if smoke_test:
        base_config, sweep_grid, max_parallel, smoke_limit = apply_smoke_overrides(base_config, sweep_grid)
        if effective_max_runs is None:
            effective_max_runs = smoke_limit
        mode_tag = "smoke"

    if dry_run:
        mode_tag = f"{mode_tag}_dry"

    sweep_name = f"{structure}_master_eval_{mode_tag}_{datetime.now().strftime('%y%m%d_%H%M%S')}"

    sweep_root, run_config_paths, concrete_configs = prepare_sweep(
        base_config=base_config,
        sweep_grid=sweep_grid,
        sweep_name=sweep_name,
    )

    if effective_max_runs is not None:
        run_config_paths = run_config_paths[:effective_max_runs]
        concrete_configs = concrete_configs[:effective_max_runs]

    master_summary_path = sweep_root / "master_summary.json"
    master_csv_path = sweep_root / "master_summary.csv"
    best_model_path = sweep_root / "best_model.json"

    master_summary: dict[str, Any] = {
        "sweep_name": sweep_name,
        "created_at": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "structure": structure,
        "cif_path": str(cif_path),
        "crystal_db_path": str(crystal_db_path),
        "dry_run": dry_run,
        "smoke_test": smoke_test,
        "max_parallel_trainings": max_parallel,
        "max_runs": effective_max_runs,
        "use_deploy_model_if_available": USE_DEPLOY_MODEL_IF_AVAILABLE,
        "eval_settings": deepcopy(eval_settings),
        "base_config": deepcopy(base_config),
        "sweep_grid": deepcopy(sweep_grid),
        "n_runs_total_prepared": len(concrete_configs),
        "n_runs_total_selected": len(run_config_paths),
        "training_results": [],
        "evaluation_results": [],
        "best_model": None,
    }

    command_previews = preview_run_commands(run_config_paths)
    master_summary["command_previews"] = command_previews

    save_json(master_summary, master_summary_path)

    print(f"Sweep root: {sweep_root}")
    print(f"Prepared runs: {len(concrete_configs)}")
    print(f"Selected runs: {len(run_config_paths)}")
    print(f"Mode: {'dry' if dry_run else ('smoke' if smoke_test else 'full')}")

    if dry_run:
        print("\nDry run only. No training or evaluation will be executed.\n")
        for preview in command_previews:
            print(f"Run: {preview['run_name']}")
            print(f"  run_config: {preview['run_config_path']}")
            print(f"  result_dir: {preview['result_dir']}")
            print(f"  train exists: {preview['train_file_exists']}")
            print(f"  valid exists: {preview['valid_file_exists']}")
            print(f"  command: {' '.join(preview['train_command'])}")
            print()
        print(f"Master summary JSON: {master_summary_path}")
        return

    n_total_runs = len(run_config_paths)
    n_completed = 0

    for train_result in run_sweep_parallel(
        run_config_paths=run_config_paths,
        max_parallel=max_parallel,
    ):
        
        n_completed += 1
        progress_pct = 100.0 * n_completed / n_total_runs

        print("\n" + "=" * 90)
        print(
            f"[SWEEP {n_completed:03d}/{n_total_runs:03d}] "
            f"({progress_pct:5.1f}%)"
        )
        print("=" * 90)

        master_summary["training_results"].append(train_result)
        save_json(master_summary, master_summary_path)

        run_name = train_result.get("run_name", "<unknown>")
        print(
            f"Training finished: {run_name} "
            f"| status={train_result.get('status')}"
        )

        if train_result.get("status") != "ok":
            row = build_eval_row(train_result=train_result, eval_result=None, error=train_result.get("error"))
            master_summary["evaluation_results"].append({
                "run_name": run_name,
                "status": "skipped_due_to_training_failure",
                "train_result": train_result,
            })
            append_master_csv(row, master_csv_path)
            save_json(master_summary, master_summary_path)
            continue

        try:
            model_path = choose_model_path(train_result)
            result_dir = Path(train_result["result_dir"])

            print(f"Starting evaluation for: {run_name}")
            print(f"Using model: {model_path}")

            eval_result = evaluate_model(
                model_path=model_path,
                structure=structure,
                cif_path=cif_path,
                output_dir=result_dir,
                crystal_db_path=crystal_db_path,
                device=eval_settings["device"],
                default_dtype=eval_settings["default_dtype"],
                frechet=eval_settings["frechet"],
                fmax=eval_settings["fmax"],
                calculator_mode=eval_settings["calculator_mode"],
                compare_crystal_modes=eval_settings["compare_crystal_modes"],
                crystal_hess_path=eval_settings["crystal_hess_path"],
                crystal_freq_out_path=eval_settings["crystal_freq_out_path"],
                crystal_hessian_units=eval_settings["crystal_hessian_units"],
                # reserved plugin interface
                run_phonopy=eval_settings.get("run_phonopy", False),
                phonopy_plugin=eval_settings.get("phonopy_plugin"),
                write_ref_db=eval_settings.get("write_ref_db", False),
                ref_db_path=eval_settings.get("ref_db_path"),
                run_id=run_name,
                dataset_split=results_group_dir,
                sweep_id=sweep_root.name,
            )

            eval_record = {
                "run_name": run_name,
                "status": "ok",
                "evaluated_at": now_iso(),
                "train_result": train_result,
                "eval_result": {
                    "n_imag_modes": eval_result.get("n_imag_modes"),
                    "n_ir_active_modes": eval_result.get("n_ir_active_modes"),
                    "zpe_eV": eval_result.get("zpe_eV"),
                    "crystal_comparison": {
                        "spectrum_rel_l2": eval_result.get("crystal_comparison", {}).get("spectrum_rel_l2"),
                    },
                    "ranking_metrics": eval_result.get("ranking_metrics", {}),
                    "artifacts": eval_result.get("artifacts", {}),
                "ref_db": eval_result.get("ref_db", {}),
                },
            }
            master_summary["evaluation_results"].append(eval_record)

            row = build_eval_row(train_result=train_result, eval_result=eval_result, error=None)
            append_master_csv(row, master_csv_path)

            best_model = choose_best_model(master_summary)
            master_summary["best_model"] = best_model
            save_json(master_summary, master_summary_path)
            if best_model is not None:
                save_json(best_model, best_model_path)
            print(
                f"[DONE {n_completed:03d}/{n_total_runs:03d}] "
                f"{run_name}"
            )
            if best_model is not None:
                print(f"Current best model: {best_model['run_name']}")
                print(f"Composite score   : {best_model['metrics'].get('composite_score')}")

        except Exception as exc:
            eval_record = {
                "run_name": run_name,
                "status": "failed",
                "evaluated_at": now_iso(),
                "train_result": train_result,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            master_summary["evaluation_results"].append(eval_record)

            row = build_eval_row(train_result=train_result, eval_result=None, error=repr(exc))
            append_master_csv(row, master_csv_path)

            best_model = choose_best_model(master_summary)
            master_summary["best_model"] = best_model
            save_json(master_summary, master_summary_path)
            if best_model is not None:
                save_json(best_model, best_model_path)

            print(f"Evaluation failed for: {run_name}")
            print(repr(exc))

    master_summary["best_model"] = choose_best_model(master_summary)
    master_summary["finished_at"] = now_iso()
    save_json(master_summary, master_summary_path)

    if master_summary["best_model"] is not None:
        save_json(master_summary["best_model"], best_model_path)

    print("\nAll runs processed.")
    print(f"Master summary JSON: {master_summary_path}")
    print(f"Master summary CSV : {master_csv_path}")
    print(f"Best model JSON    : {best_model_path}")

    if master_summary["best_model"] is not None:
        best = master_summary["best_model"]
        print("\nBest model identified:")
        print(f"  Run name   : {best['run_name']}")
        print(f"  Model path : {best['model_path']}")
        print(f"  Result dir : {best['result_dir']}")
        print("  Metrics:")
        for k, v in best["metrics"].items():
            print(f"    {k}: {v}")
    else:
        print("\nNo successful evaluated model was available to rank.")


if __name__ == "__main__":
    '''
    python run_master.py --config configs/master_cfg/AlN_10_90.yaml

    Eval-only:
    python run_master.py \
    --config configs/AlN_10_90.yaml \
    --eval-only \
    --sweep-dir AlN_10_90/AlN_master_eval_full_260504_120000

    
    Eval-only dry run:
    python run_master.py \
    --config configs/AlN_10_90.yaml \
    --eval-only \
    --sweep-dir AlN_10_90/AlN_master_eval_full_260504_120000 \
    --max-eval-runs 3 \
    --dry-run
    '''
    main()
