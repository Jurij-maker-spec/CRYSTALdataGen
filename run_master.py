#!/usr/bin/env python3
"""
run_master.py

Top-level orchestration script for:
1. preparing a hyperparameter sweep from config
2. optionally dry-running the sweep setup
4. running training jobs in parallel
5. starting evaluation immediately when each training job finishes
6. saving a sweep-wide summary
7. explicitly naming the best model after the sweep
"""

from __future__ import annotations

import json
import traceback
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
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

from util.io import ensure_dir, load_json, load_yaml, save_json
from util.eval import (
    append_master_csv,
    build_eval_row,
    choose_best_model,
    choose_model_path,
    eval_cif_structure_name,
    is_eval_complete,
    make_compact_eval_record,
    now_iso,
    run_single_model_eval,
)
from util.common_helpers import require_path

# ============================================================
# USER CONFIG
# ============================================================
USE_DEPLOY_MODEL_IF_AVAILABLE = False


# ============================================================
# HELPERS
# ============================================================

def rebase_run_config_for_current_machine(
    run_config_path: Path,
    *,
    restart_latest: bool,
) -> None:
    run_dir = run_config_path.parent
    run_cfg = load_json(run_config_path)

    run_cfg["project_root"] = str(PROJECT_ROOT)
    run_cfg["data_dir"] = str(PROJECT_ROOT / "data")
    run_cfg["deploy_root"] = str(PROJECT_ROOT / "models")
    run_cfg["run_dir"] = str(run_dir)
    run_cfg["restart_latest"] = restart_latest

    save_json(run_cfg, run_config_path)


def checkpoint_exists(run_dir: Path) -> bool:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return False

    patterns = ["*.pt", "*.pth", "*.model", "*.ckpt"]
    return any(checkpoint_dir.glob(pattern) for pattern in patterns)


def finished_model_exists(run_dir: Path) -> bool:
    models_dir = run_dir / "models"
    return models_dir.exists() and any(models_dir.glob("*.model"))


def latest_model_path(run_dir: Path) -> Path:
    models = sorted((run_dir / "models").glob("*.model"))
    if not models:
        raise FileNotFoundError(f"No finished .model found in {run_dir}")
    return models[-1]


def enable_restart_latest(run_config_path: Path) -> None:
    run_cfg = load_json(run_config_path)
    run_cfg["restart_latest"] = True
    save_json(run_cfg, run_config_path)


def train_result_from_finished_run(run_config_path: Path) -> dict[str, Any]:
    run_dir = run_config_path.parent
    run_cfg = load_json(run_config_path)
    run_name = run_cfg.get("run_name", run_dir.name)

    return {
        "run_name": run_name,
        "status": "ok",
        "result_dir": str(run_dir),
        "run_config_path": str(run_config_path),
        "trained_model_path": str(latest_model_path(run_dir)),
        "deploy_model_path": None,
        "error": None,
    }


def evaluate_train_result_and_update_summary(
    *,
    train_result: dict[str, Any],
    structure: str,
    cif_path: Path,
    crystal_db_path: Path,
    eval_settings: dict[str, Any],
    results_group_dir: str,
    sweep_root: Path,
    master_summary: dict[str, Any],
    master_summary_path: Path,
    master_csv_path: Path,
    best_model_path: Path,
) -> None:
    run_name = train_result.get("run_name", "<unknown>")

    try:
        print(f"Starting evaluation for: {run_name}")

        eval_result = run_single_model_eval(
            evaluate_model_func=evaluate_model,
            train_result=train_result,
            structure=structure,
            cif_path=cif_path,
            crystal_db_path=crystal_db_path,
            eval_settings=eval_settings,
            dataset_split=results_group_dir,
            sweep_id=sweep_root.name,
            use_deploy_model_if_available=USE_DEPLOY_MODEL_IF_AVAILABLE,
        )

        eval_record = make_compact_eval_record(run_name, train_result, eval_result)
        master_summary["evaluation_results"].append(eval_record)

        row = build_eval_row(train_result=train_result, eval_result=eval_result, error=None)
        append_master_csv(row, master_csv_path)

        best_model = choose_best_model(master_summary)
        master_summary["best_model"] = best_model
        save_json(master_summary, master_summary_path)

        if best_model is not None:
            save_json(best_model, best_model_path)

        print(f"Evaluation finished for: {run_name}")

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


def resolve_resume_sweep_dir(value: str | Path) -> Path:
    p = Path(value)

    if p.is_absolute():
        resolved = p
    else:
        resolved = PROJECT_ROOT / p

    if not resolved.exists():
        raise FileNotFoundError(f"Missing resume sweep directory: {resolved}")

    if not resolved.is_dir():
        raise NotADirectoryError(f"Resume path is not a directory: {resolved}")

    return resolved


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


def resolve_results_group_root(group_dir: str | None) -> Path:
    results_root = PROJECT_ROOT / "results"

    if group_dir is None or str(group_dir).strip() == "":
        return results_root

    return results_root / group_dir


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

# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a MACELES hyperparameter sweep."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file. Relative paths are resolved under configs/master_cfg.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing sweep from cfg['resume_sweep_dir'].",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview generated run configs and commands without training/evaluation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_root = resolve_config_path(args.config)
    cfg = load_yaml(config_root)

    # ============================================================
    # Check correct environment
    # ============================================================
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env != REQUIRED_ENV:
        raise RuntimeError(
            f"This script must be run in conda env '{REQUIRED_ENV}', "
            f"but current env is '{current_env}'."
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
        cfg.get("cif_path", f"inference/CIFs/{eval_cif_structure}.cif"),
        base=PROJECT_ROOT
    )
    crystal_db_path = require_path(
        cfg["crystal_db_path"],
        base=PROJECT_ROOT,
    )
    dry_run = args.dry_run or cfg.get("dry_run", False)
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

    if not cif_path.exists():
        raise FileNotFoundError(f"Missing CIF: {cif_path}")
    if not crystal_db_path.exists():
        raise FileNotFoundError(f"Missing CRYSTAL DB: {crystal_db_path}")

    group_root = resolve_results_group_root(results_group_dir)
    ensure_dir(group_root)
    base_config["results_root"] = str(group_root)

    sweep_grid = deepcopy(sweep_grid)
    dataset_file_sweep = cfg.get("dataset_file_sweep")
    max_parallel = max_parallel_trainings
    effective_max_runs = max_runs

    mode_tag = "full"

    if dry_run:
        mode_tag = f"{mode_tag}_dry"


    # ============================================================
    # Handling cases block
    # ============================================================
    if args.resume:
        resume_sweep_dir = cfg.get("resume_sweep_dir")
        if not resume_sweep_dir:
            raise ValueError("--resume requires 'resume_sweep_dir' in the YAML config.")

        sweep_root = resolve_resume_sweep_dir(resume_sweep_dir)
        sweep_name = sweep_root.name

        run_config_paths = sorted(sweep_root.glob("*/run_config.json"))

        if not run_config_paths:
            raise FileNotFoundError(f"No run_config.json files found under: {sweep_root}")

        concrete_configs = [load_json(p) for p in run_config_paths]

    else:
        sweep_name = f"{structure}_master_eval_{mode_tag}_{datetime.now().strftime('%y%m%d_%H%M%S')}"

        sweep_root, run_config_paths, concrete_configs = prepare_sweep(
            base_config=base_config,
            sweep_grid=sweep_grid,
            sweep_name=sweep_name,
            dataset_file_sweep=dataset_file_sweep,
        )

    if effective_max_runs is not None and not args.resume:
        run_config_paths = run_config_paths[:effective_max_runs]
        concrete_configs = concrete_configs[:effective_max_runs]

    master_summary_path = sweep_root / "master_summary.json"
    master_csv_path = sweep_root / "master_summary.csv"
    best_model_path = sweep_root / "best_model.json"

    if args.resume and master_summary_path.exists():
        master_summary = load_json(master_summary_path)
        master_summary.setdefault("training_results", [])
        master_summary.setdefault("evaluation_results", [])
    else:
        master_summary: dict[str, Any] = {
            "sweep_name": sweep_name,
            "dataset_file_sweep": deepcopy(dataset_file_sweep),
            "created_at": now_iso(),
            "project_root": str(PROJECT_ROOT),
            "structure": structure,
            "cif_path": str(cif_path),
            "crystal_db_path": str(crystal_db_path),
            "dry_run": dry_run,
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
    print(f"Mode: {'dry' if dry_run else 'full'}")

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

    # ============================================================
    # Resume block
    # ============================================================
    if args.resume:
        print(f"Resume mode: {sweep_root}")
        print(f"Run configs found: {len(run_config_paths)}")

        already_done: list[Path] = []
        eval_pending: list[Path] = []
        train_pending: list[Path] = []
        fresh_train_pending: list[Path] = []

        for run_config_path in run_config_paths:
            run_dir = run_config_path.parent
            run_cfg = load_json(run_config_path)
            run_name = run_cfg.get("run_name", run_dir.name)

            if is_eval_complete(run_dir, structure):
                already_done.append(run_config_path)
                print(f"[SKIP] Already evaluated: {run_name}")
                continue

            if finished_model_exists(run_dir):
                eval_pending.append(run_config_path)
                print(f"[EVAL] Finished model found, evaluation missing: {run_name}")
                continue

            if checkpoint_exists(run_dir):
                rebase_run_config_for_current_machine(
                    run_config_path,
                    restart_latest=True,
                )
                train_pending.append(run_config_path)
                print(f"[TRAIN-RESUME] Checkpoint found: {run_name}")
                continue

            # No model and no checkpoint means the run was prepared but never started.
            # Start it normally.
            rebase_run_config_for_current_machine(
                run_config_path,
                restart_latest=False,
            )

            train_pending.append(run_config_path)
            fresh_train_pending.append(run_config_path)
            print(f"[TRAIN-START] No checkpoint found, starting fresh: {run_name}")

        master_summary["resume_summary"] = {
            "resumed_at": now_iso(),
            "already_done": len(already_done),
            "eval_pending": len(eval_pending),
            "train_pending": len(train_pending),
            "fresh_train_pending": len(fresh_train_pending),
            "checkpoint_train_pending": len(train_pending) - len(fresh_train_pending),
        }
        save_json(master_summary, master_summary_path)

        print("\nResume classification:")
        print(f"  already evaluated        : {len(already_done)}")
        print(f"  eval pending             : {len(eval_pending)}")
        print(f"  train pending            : {len(train_pending)}")
        print(f"  fresh train pending      : {len(fresh_train_pending)}")
        print(f"  checkpoint train pending : {len(train_pending) - len(fresh_train_pending)}")

        # 1. First evaluate models that already finished training.
        n_eval_pending = len(eval_pending)
        for i, run_config_path in enumerate(eval_pending, start=1):
            progress_pct = 100.0 * i / n_eval_pending if n_eval_pending else 100.0

            print("\n" + "=" * 90)
            print(f"[RESUME-EVAL {i:03d}/{n_eval_pending:03d}] ({progress_pct:5.1f}%)")
            print("=" * 90)
            train_result = train_result_from_finished_run(run_config_path)
            master_summary["training_results"].append(train_result)
            save_json(master_summary, master_summary_path)

            evaluate_train_result_and_update_summary(
                train_result=train_result,
                structure=structure,
                cif_path=cif_path,
                crystal_db_path=crystal_db_path,
                eval_settings=eval_settings,
                results_group_dir=results_group_dir,
                sweep_root=sweep_root,
                master_summary=master_summary,
                master_summary_path=master_summary_path,
                master_csv_path=master_csv_path,
                best_model_path=best_model_path,
            )

        # 2. Then restart interrupted trainings from latest checkpoint.
        if train_pending:
            print("\nStarting/resuming pending trainings...")
            n_train_pending = len(train_pending)
            n_train_completed = 0
            for train_result in run_sweep_parallel(
                run_config_paths=train_pending,
                max_parallel=max_parallel,
            ):
                n_train_completed += 1
                progress_pct = 100.0 * n_train_completed / n_train_pending
                print("\n" + "=" * 90)
                print(
                    f"[RESUME-TRAIN {n_train_completed:03d}/{n_train_pending:03d}] "
                    f"({progress_pct:5.1f}%)"
                )
                print("=" * 90)

                master_summary["training_results"].append(train_result)
                save_json(master_summary, master_summary_path)

                run_name = train_result.get("run_name", "<unknown>")
                print(f"Resumed training finished: {run_name} | status={train_result.get('status')}")
                if train_result.get("status") != "ok":
                    print(f"Training error: {train_result.get('error')}")

                if train_result.get("status") != "ok":
                    row = build_eval_row(
                        train_result=train_result,
                        eval_result=None,
                        error=train_result.get("error"),
                    )
                    master_summary["evaluation_results"].append({
                        "run_name": run_name,
                        "status": "skipped_due_to_training_failure",
                        "train_result": train_result,
                    })
                    append_master_csv(row, master_csv_path)
                    save_json(master_summary, master_summary_path)
                    continue

                evaluate_train_result_and_update_summary(
                    train_result=train_result,
                    structure=structure,
                    cif_path=cif_path,
                    crystal_db_path=crystal_db_path,
                    eval_settings=eval_settings,
                    results_group_dir=results_group_dir,
                    sweep_root=sweep_root,
                    master_summary=master_summary,
                    master_summary_path=master_summary_path,
                    master_csv_path=master_csv_path,
                    best_model_path=best_model_path,
                )

        master_summary["best_model"] = choose_best_model(master_summary)
        master_summary["finished_at"] = now_iso()
        save_json(master_summary, master_summary_path)

        if master_summary["best_model"] is not None:
            save_json(master_summary["best_model"], best_model_path)

        print("\nResume pass finished.")
        print(f"Master summary JSON: {master_summary_path}")
        print(f"Master summary CSV : {master_csv_path}")
        print(f"Best model JSON    : {best_model_path}")
        return

    # ============================================================
    # Main block
    # ============================================================
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

        evaluate_train_result_and_update_summary(
            train_result=train_result,
            structure=structure,
            cif_path=cif_path,
            crystal_db_path=crystal_db_path,
            eval_settings=eval_settings,
            results_group_dir=results_group_dir,
            sweep_root=sweep_root,
            master_summary=master_summary,
            master_summary_path=master_summary_path,
            master_csv_path=master_csv_path,
            best_model_path=best_model_path,
        )

        print(
            f"[DONE {n_completed:03d}/{n_total_runs:03d}] "
            f"{run_name}"
        )

        if master_summary.get("best_model") is not None:
            best_model = master_summary["best_model"]
            print(f"Current best model: {best_model['run_name']}")
            print(f"Composite score   : {best_model['metrics'].get('composite_score')}")

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
    '''
    main()
