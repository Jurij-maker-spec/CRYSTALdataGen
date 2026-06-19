#!/usr/bin/env python3
"""
sweep_training.py

Generate and run a parallel hyperparameter sweep for MACE / MACELES.

This module is designed to be:
- imported by run_master_eval.py
- runnable directly for training-only sweeps

Behavior
--------
- generates one run_config.json per concrete run
- stores them under a sweep directory
- runs up to max_parallel training jobs
- returns finished training results one by one
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# Allow importing sibling module train_model.py
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_model import run_training_from_config


# ============================================================
# USER BASE CONFIG
# ============================================================

PROJECT_ROOT = THIS_DIR

BASE_CONFIG: dict[str, Any] = {
    # ----- data -----
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(PROJECT_ROOT / "data"),
    "train_file": "train_SiO2.xyz",
    "valid_file": "valid_interp_SiO2.xyz",

    # ----- model / chemistry -----
    "model_type": "MACELES",
    "atomic_numbers": [8, 14],
    "chem": "SiO2",
    "foundation_model": None,
    "E0s": "average",

    # ----- training -----
    "batch_size": 2,
    "valid_batch_size": 2,
    "max_epochs": 300,
    "energy_weight": 1.0,
    "forces_weight": 100.0,
    "r_max": 6.5,
    "use_stress": False,
    "stress_weight": 2.0,
    "default_dtype": "float64",
    "device": "cuda",
    "num_workers": 0,
    "use_ema": True,
    "use_swa": False,
    "restart_latest": False,
    "seed": 0,

    # ----- keys -----
    "energy_key": "energy",
    "forces_key": "forces",
    "stress_key": "stress",

    # ----- environment -----
    "env": {
        "OMP_NUM_THREADS": "1",
    },

    # ----- output layout -----
    "results_root": str(PROJECT_ROOT / "results"),
    "deploy_root": str(PROJECT_ROOT / "models"),
    "extract_head": "default",

    # ----- behavior -----
    "run_extraction": True,
    "overwrite_existing_logs": True,
}

# Default sweep grid. Adjust as needed.
DEFAULT_SWEEP_GRID: dict[str, list[Any]] = {
    "seed": [1, 2],
    "r_max": [3.0, 4.0, 5.0],
    "batch_size": [2],
    "valid_batch_size": [2],
    "energy_weight": [2.0, 1.0],
    "forces_weight": [100.0],
    "use_stress": [False],
    "stress_weight": [2.0],
}


# ============================================================
# HELPERS
# ============================================================

def now_tag() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_float_for_name(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{value:.6g}"
    return text.replace(".", "p")


def les_tag_from_path(value: str | Path | None) -> str | None:
    if value is None:
        return None

    stem = Path(str(value)).stem
    stem = stem.replace("les_", "")
    stem = stem.replace("_", "")

    if stem.startswith("dl"):
        return f"les{stem}"

    return f"les{stem}"


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
    les_tag = les_tag_from_path(cfg.get("les_arguments"))
    if les_tag is not None:
        bits.append(les_tag)

    if cfg.get("dataset_tag") is not None:
        bits.append(str(cfg["dataset_tag"]))

    if cfg.get("use_stress", False):
        bits.append(f"sw{format_float_for_name(cfg['stress_weight'])}")

    return "_".join(bits)


def normalized_dataset_variants(dataset_file_sweep: dict[str, Any] | None):
    if not dataset_file_sweep or not dataset_file_sweep.get("enabled", False):
        return [None]

    datasets = dataset_file_sweep.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("dataset_file_sweep.enabled=true requires non-empty datasets dict")

    variants = []
    for tag, data in datasets.items():
        if "train_file" not in data or "valid_file" not in data:
            raise ValueError(f"Dataset variant {tag} needs train_file and valid_file")

        variants.append({
            "dataset_tag": str(tag),
            "dataset_size": data.get("dataset_size"),
            "train_file": str(data["train_file"]),
            "valid_file": str(data["valid_file"]),
        })

    return variants


def generate_sweep_configs(
    base_cfg: dict[str, Any],
    sweep_grid: dict[str, list[Any]],
    dataset_file_sweep: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    keys = list(sweep_grid.keys())
    value_lists = [sweep_grid[k] for k in keys]
    dataset_variants = normalized_dataset_variants(dataset_file_sweep)

    configs: list[dict[str, Any]] = []

    for values in itertools.product(*value_lists):
        for dataset_variant in dataset_variants:
            cfg = deepcopy(base_cfg)

            for key, value in zip(keys, values):
                cfg[key] = value

            if dataset_variant is not None:
                cfg.update(dataset_variant)
            
            if "train_file" not in cfg or "valid_file" not in cfg:
                raise ValueError(
                    "No train/valid files defined. "
                    "Use dataset_file_sweep or define train_file/valid_file in base_config."
                )

            cfg["run_name"] = make_run_name(cfg)
            configs.append(cfg)

    return configs


def write_run_configs(
    configs: list[dict[str, Any]],
    sweep_root: Path,
) -> list[Path]:
    ensure_dir(sweep_root)
    run_config_paths: list[Path] = []

    for i, cfg in enumerate(configs, start=1):
        run_dir = sweep_root / cfg["run_name"]
        cfg["run_index"] = i
        cfg["sweep_root"] = str(sweep_root)
        cfg["run_dir"] = str(run_dir)

        ensure_dir(run_dir)
        run_config_path = run_dir / "run_config.json"
        save_json(cfg, run_config_path)
        run_config_paths.append(run_config_path)

    return run_config_paths


def write_sweep_manifest(
    configs: list[dict[str, Any]],
    sweep_root: Path,
    sweep_name: str,
) -> Path:
    
    ensure_dir(sweep_root)
    manifest = {
        "sweep_name": sweep_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_runs": len(configs),
        "runs": [
            {
                "run_name": cfg["run_name"],
                "run_dir": cfg["run_dir"],
                "seed": cfg["seed"],
                "r_max": cfg["r_max"],
                "batch_size": cfg["batch_size"],
                "valid_batch_size": cfg["valid_batch_size"],
                "energy_weight": cfg["energy_weight"],
                "forces_weight": cfg["forces_weight"],
                "use_stress": cfg["use_stress"],
                "stress_weight": cfg["stress_weight"],
                "les_arguments": cfg.get("les_arguments"),
            }
            for cfg in configs
        ],
    }

    manifest_path = sweep_root / "sweep_manifest.json"
    save_json(manifest, manifest_path)
    return manifest_path


# ============================================================
# PUBLIC API
# ============================================================

def prepare_sweep(
    base_config: dict[str, Any] | None = None,
    sweep_grid: dict[str, list[Any]] | None = None,
    sweep_name: str | None = None,
    dataset_file_sweep: dict[str, Any] | None = None,
) -> tuple[Path, list[Path], list[dict[str, Any]]]:
    """
    Create all run configs and return:
    - sweep_root
    - run_config_paths
    - concrete configs
    """
    if base_config is None:
        base_config = deepcopy(BASE_CONFIG)
    else:
        base_config = deepcopy(base_config)

    if sweep_grid is None:
        sweep_grid = deepcopy(DEFAULT_SWEEP_GRID)
    else:
        sweep_grid = deepcopy(sweep_grid)

    if sweep_name is None:
        sweep_name = f"{base_config['chem']}_sweep_{now_tag()}"

    sweep_root = Path(base_config["results_root"]) / sweep_name
    
    ensure_dir(sweep_root)

    configs = generate_sweep_configs(
        base_cfg=base_config,
        sweep_grid=sweep_grid,
        dataset_file_sweep=dataset_file_sweep,
    )

    run_config_paths = write_run_configs(configs, sweep_root)
    write_sweep_manifest(configs, sweep_root, sweep_name)

    return sweep_root, run_config_paths, configs


def run_sweep_parallel(
    run_config_paths: list[str | Path],
    max_parallel: int = 4,
) -> Iterator[dict[str, Any]]:
    """
    Run training jobs in parallel and yield one completed result at a time.

    This is the key function for run_master_eval.py:
    as soon as one training run finishes, it yields the result dict.
    """
    run_config_paths = [Path(p).resolve() for p in run_config_paths]

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        future_to_cfg = {
            executor.submit(run_training_from_config, cfg_path): cfg_path
            for cfg_path in run_config_paths
        }

        for future in as_completed(future_to_cfg):
            cfg_path = future_to_cfg[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "phase": "executor_exception",
                    "run_config_path": str(cfg_path),
                    "error": repr(exc),
                }
            yield result


def run_and_collect_sweep(
    run_config_paths: list[str | Path],
    max_parallel: int = 4,
    sweep_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience wrapper for training-only use.
    """
    results: list[dict[str, Any]] = []

    for result in run_sweep_parallel(run_config_paths=run_config_paths, max_parallel=max_parallel):
        results.append(result)

        if sweep_root is not None:
            sweep_root = Path(sweep_root)
            ensure_dir(sweep_root)
            save_json({"results": results}, sweep_root / "training_results.json")

    return results


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sweep_root, run_config_paths, configs = prepare_sweep()

    if args.max_runs is not None:
        run_config_paths = run_config_paths[:args.max_runs]
        configs = configs[:args.max_runs]

    print(f"Sweep root: {sweep_root}")
    print(f"Prepared {len(run_config_paths)} runs")

    if args.prepare_only:
        return

    results = run_and_collect_sweep(
        run_config_paths=run_config_paths,
        max_parallel=args.max_parallel,
        sweep_root=sweep_root,
    )

    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
