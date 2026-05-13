#!/usr/bin/env python3
"""
train_model.py

Run a single MACE / MACELES training job from one JSON config.

Usage
-----
python 2_training/train_model.py --run-config /path/to/run_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# HELPERS
# ============================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def load_run_config(run_config_path: Path) -> dict[str, Any]:
    with open(run_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def save_json(payload: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2, sort_keys=True)


def format_float_for_name(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{value:.6g}"
    return text.replace(".", "p")


def make_run_name(cfg: dict[str, Any]) -> str:
    bits = [
        cfg["model_type"],
        f"bs{cfg['batch_size']}",
        f"ep{cfg['max_epochs']}",
        f"ew{format_float_for_name(cfg['energy_weight'])}",
        f"fw{format_float_for_name(cfg['forces_weight'])}",
        f"rmax{format_float_for_name(cfg['r_max'])}",
        f"seed{cfg['seed']}",
        cfg["chem"],
    ]
    if cfg.get("use_stress", False):
        bits.append(f"sw{format_float_for_name(cfg['stress_weight'])}")
    return "_".join(bits)


def build_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    project_root = Path(cfg["project_root"]).resolve()
    data_dir = Path(cfg["data_dir"]).resolve()
    results_root = Path(cfg["results_root"]).resolve()
    deploy_root = Path(cfg["deploy_root"]).resolve()

    train_file = data_dir / cfg["train_file"]
    valid_file = data_dir / cfg["valid_file"]

    run_name = cfg["run_name"]
    result_dir = Path(cfg.get("run_dir", results_root / run_name)).resolve()
    model_dir = result_dir / "models"
    checkpoint_dir = result_dir / "checkpoints"
    # deploy_path = deploy_root / f"{run_name}_sh.model"
    deploy_path = result_dir / "models" / f"{run_name}.model"

    return {
        "project_root": project_root,
        "data_dir": data_dir,
        "results_root": results_root,
        "deploy_root": deploy_root,
        "train_file": train_file,
        "valid_file": valid_file,
        "result_dir": result_dir,
        "model_dir": model_dir,
        "checkpoint_dir": checkpoint_dir,
        "deploy_path": deploy_path,
        "train_log": result_dir / "train.log",
        "extract_log": result_dir / "extract.log",
        "cmd_txt": result_dir / "train_command.txt",
        "status_json": result_dir / "train_status.json",
        "result_json": result_dir / "train_result.json",
        "run_json_copy": result_dir / "run_config.json",
    }


def prepare_directories(paths: dict[str, Path]) -> None:
    ensure_dir(paths["result_dir"])
    ensure_dir(paths["model_dir"])
    ensure_dir(paths["checkpoint_dir"])
    ensure_dir(paths["deploy_root"])


def apply_environment(cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in cfg.get("env", {}).items():
        env[str(key)] = str(value)
    return env


def build_train_command(cfg: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", cfg["run_name"],
        "--model", cfg["model_type"],
        "--train_file", str(paths["train_file"]),
        "--valid_file", str(paths["valid_file"]),
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
        "--work_dir", str(paths["result_dir"]),
        "--log_dir", str(paths["result_dir"]),
        "--model_dir", str(paths["model_dir"]),
        "--checkpoints_dir", str(paths["checkpoint_dir"]),
        "--results_dir", str(paths["result_dir"]),
        "--seed", str(cfg["seed"]),
        "--r_max", str(cfg["r_max"]),
    ]

    foundation_model = cfg.get("foundation_model")
    if foundation_model is not None:
        cmd += ["--foundation_model", str(foundation_model)]

    if cfg.get("use_stress", False):
        cmd += [
            "--stress_key", cfg["stress_key"],
            "--stress_weight", str(cfg["stress_weight"]),
        ]
    else:
        cmd += ["--stress_weight", "0.0"]

    if cfg.get("restart_latest", False):
        cmd.append("--restart_latest")
    if cfg.get("use_ema", False):
        cmd.append("--ema")
    if cfg.get("use_swa", False):
        cmd.append("--swa")

    return cmd


def build_extract_command(cfg: dict[str, Any], trained_model_path: Path, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable, "-m", "mace.cli.mace_select_head",
        "--model", str(trained_model_path),
        "--head", cfg["extract_head"],
        "--output", str(paths["deploy_path"]),
    ]


def locate_trained_model(cfg: dict[str, Any], paths: dict[str, Path]) -> Path:
    candidate_main = paths["result_dir"] / f"{cfg['run_name']}.model"
    candidate_model_dir = paths["model_dir"] / f"{cfg['run_name']}.model"

    if candidate_main.exists():
        return candidate_main
    if candidate_model_dir.exists():
        return candidate_model_dir

    candidates = sorted(paths["model_dir"].glob("*.model"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .model file found in {paths['model_dir']}")
    return candidates[-1]


def write_command_text(train_cmd: list[str], paths: dict[str, Path]) -> None:
    with open(paths["cmd_txt"], "w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(x) for x in train_cmd) + "\n")


def run_subprocess(
    cmd: list[str],
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
    overwrite: bool = True,
) -> subprocess.CompletedProcess:
    mode = "w" if overwrite else "a"
    with open(log_path, mode, encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            env=env,
            check=False,
        )
    return result


# ============================================================
# CORE
# ============================================================

def run_training_from_config(run_config_path: str | Path) -> dict[str, Any]:
    run_config_path = Path(run_config_path).resolve()
    cfg = load_run_config(run_config_path)
    cfg = deepcopy(cfg)

    if not cfg.get("run_name"):
        cfg["run_name"] = make_run_name(cfg)

    paths = build_paths(cfg)
    prepare_directories(paths)

    ensure_file_exists(paths["train_file"], "training file")
    ensure_file_exists(paths["valid_file"], "validation file")

    # Keep a copy of the config inside the run directory
    save_json(cfg, paths["run_json_copy"])

    train_cmd = build_train_command(cfg, paths)
    write_command_text(train_cmd, paths)

    status = {
        "status": "running",
        "phase": "training",
        "run_name": cfg["run_name"],
        "run_config_path": run_config_path,
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }
    save_json(status, paths["status_json"])

    env = apply_environment(cfg)

    try:
        train_result = run_subprocess(
            cmd=train_cmd,
            log_path=paths["train_log"],
            cwd=paths["result_dir"],
            env=env,
            overwrite=cfg.get("overwrite_existing_logs", True),
        )

        if train_result.returncode != 0:
            result = {
                "status": "failed",
                "phase": "training",
                "run_name": cfg["run_name"],
                "run_config_path": run_config_path,
                "result_dir": paths["result_dir"],
                "train_log": paths["train_log"],
                "returncode": train_result.returncode,
                "finished_at": now_iso(),
            }
            save_json(result, paths["status_json"])
            save_json(result, paths["result_json"])
            return result

        trained_model_path = locate_trained_model(cfg, paths)

        deploy_model_path = None
        extract_returncode = None

        if cfg.get("run_extraction", True):
            status.update({
                "phase": "extracting_head",
                "updated_at": now_iso(),
            })
            save_json(status, paths["status_json"])

            extract_cmd = build_extract_command(cfg, trained_model_path, paths)
            extract_result = run_subprocess(
                cmd=extract_cmd,
                log_path=paths["extract_log"],
                cwd=paths["result_dir"],
                env=env,
                overwrite=cfg.get("overwrite_existing_logs", True),
            )
            extract_returncode = extract_result.returncode

            if extract_result.returncode == 0 and paths["deploy_path"].exists():
                deploy_model_path = paths["deploy_path"]
            else:
                deploy_model_path = None

        result = {
            "status": "ok",
            "phase": "done",
            "run_name": cfg["run_name"],
            "run_config_path": run_config_path,
            "result_dir": paths["result_dir"],
            "model_dir": paths["model_dir"],
            "checkpoint_dir": paths["checkpoint_dir"],
            "train_log": paths["train_log"],
            "extract_log": paths["extract_log"],
            "trained_model_path": trained_model_path,
            "deploy_model_path": deploy_model_path,
            "train_returncode": train_result.returncode,
            "extract_returncode": extract_returncode,
            "finished_at": now_iso(),
        }

        save_json(result, paths["status_json"])
        save_json(result, paths["result_json"])
        return result

    except Exception as exc:
        result = {
            "status": "failed",
            "phase": "exception",
            "run_name": cfg["run_name"],
            "run_config_path": run_config_path,
            "result_dir": paths["result_dir"],
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "finished_at": now_iso(),
        }
        save_json(result, paths["status_json"])
        save_json(result, paths["result_json"])
        return result


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True, help="Path to run_config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training_from_config(args.run_config)
    print(json.dumps(to_serializable(result), indent=2))


if __name__ == "__main__":
    main()

