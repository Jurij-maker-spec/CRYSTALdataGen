#!/usr/bin/env python3
"""
rerun_results.py

Recursively scan results/ and rerun plotting/evaluation for trained runs.

Default:
    scans PROJECT_ROOT/results

Examples:
    python rerun_results.py --plot-only
    python rerun_results.py --eval-missing
    python rerun_results.py --force-reeval
    python rerun_results.py --dry-run
    python rerun_results.py --structure SiO2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT_DEFAULT = PROJECT_ROOT / "results"
INFERENCE_DIR = PROJECT_ROOT / "inference"
TRAINING_DIR = PROJECT_ROOT / "training"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from inference import evaluate_model
from util.eval import build_eval_row, append_master_csv, choose_best_model
from util.io import load_json, save_json
from util.eval import eval_cif_structure_name
from util.ref_db import update_model_ranking_metrics, read_model_evaluation
from inference import make_composite_score


REQUIRED_ENV = "mace_env_3_12"

STRUCTURE_ALIASES = {
    "TiO2": "TiO2_rutil",
}

# ============================================================
# discovery helpers
# ============================================================

def check_conda_env() -> None:
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env != REQUIRED_ENV:
        raise RuntimeError(
            f"Run this in conda env '{REQUIRED_ENV}', current env is '{env}'."
        )


def normalize_structure_name(structure: str) -> str:
    return STRUCTURE_ALIASES.get(structure, structure)


def has_model(run_dir: Path) -> bool:
    return (
        (run_dir / "models").exists()
        and any((run_dir / "models").glob("*.model"))
    )


def latest_model_path(run_dir: Path) -> Path:
    models = sorted((run_dir / "models").glob("*.model"))
    if not models:
        raise FileNotFoundError(f"No model found in {run_dir}")
    return models[-1]


def is_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "run_config.json").exists()
        and has_model(path)
    )


def discover_run_dirs(results_root: Path) -> list[Path]:
    return sorted(
        p for p in results_root.rglob("*")
        if is_run_dir(p)
    )


def normalize_structure_name(name: str) -> str:
    return STRUCTURE_ALIASES.get(name, name)


def extract_structure_from_dataset_filename(filename: str) -> str | None:
    """
    Examples:
        sc_train.xyz_TiO2_rutil_PBE_495.xyz -> TiO2_rutil_PBE
        sc_valid.xyz_SiO2_PBE_55.xyz       -> SiO2_PBE
    """
    stem = Path(filename).name

    m = re.search(r"\.xyz_(.+?)_\d+\.xyz$", stem)
    if m:
        return normalize_structure_name(m.group(1))

    return None


def extract_structure_from_results_group(name: str) -> str | None:
    """
    Examples:
        TiO2_rutil_PBE_sc      -> TiO2_rutil_PBE
        TiO2_rutil_10_90_1000  -> TiO2_rutil
        SiO2_PBE_1000_10_90    -> SiO2_PBE
        AlN_PBE_sc             -> AlN_PBE
        Al2O3_10_90            -> Al2O3
    """
    tokens = name.split("_")

    # Remove known dataset/split suffix tokens.
    remove_suffixes = {"sc"}
    while tokens and (tokens[-1] in remove_suffixes or tokens[-1].isdigit()):
        tokens.pop()

    if not tokens:
        return None

    return normalize_structure_name("_".join(tokens))


def infer_structure(run_dir: Path, run_cfg: dict[str, Any]) -> str:
    # 1. Most reliable: train/valid filenames contain the real DB structure.
    for key in ("train_file", "valid_file"):
        value = run_cfg.get(key)
        if value:
            structure = extract_structure_from_dataset_filename(str(value))
            if structure:
                return structure

    # 2. Next: results_root, e.g. results/TiO2_rutil_PBE_sc
    results_root = run_cfg.get("results_root")
    if results_root:
        structure = extract_structure_from_results_group(Path(results_root).name)
        if structure:
            return structure

    # 3. Next: sweep_root, e.g. TiO2_rutil_PBE_master_eval_full_...
    sweep_root = run_cfg.get("sweep_root")
    if sweep_root:
        sweep_name = Path(sweep_root).name
        marker = "_master_eval_"
        if marker in sweep_name:
            structure = sweep_name.split(marker)[0]
            return normalize_structure_name(structure)

    # 4. Next: explicit structure field, if present.
    if "structure" in run_cfg:
        return normalize_structure_name(str(run_cfg["structure"]))

    # 5. Parent split directory, e.g. results/SiO2_10_90
    try:
        split_name = run_dir.parents[1].name
        structure = extract_structure_from_results_group(split_name)
        if structure:
            return structure
    except Exception:
        pass

    # 6. Last fallback: run_name. This may only say TiO2, so use cautiously.
    run_name = str(run_cfg.get("run_name", run_dir.name))
    m = re.search(r"_([A-Z][A-Za-z0-9]*[0-9]*(?:_[A-Za-z0-9]+)*)$", run_name)
    if m:
        return normalize_structure_name(m.group(1))

    raise ValueError(f"Could not infer structure for run_dir={run_dir}")


def infer_dataset_split(run_dir: Path) -> str:
    # results/<split>/<sweep>/<run>
    try:
        return run_dir.parents[1].name
    except IndexError:
        return "ungrouped"


def infer_sweep_id(run_dir: Path) -> str:
    try:
        return run_dir.parent.name
    except Exception:
        return "manual"


def default_cif_path(structure: str) -> Path:
    structure = normalize_structure_name(structure)
    eval_structure = eval_cif_structure_name(structure)
    return PROJECT_ROOT / "inference" / "CIFs" / f"{eval_structure}.cif"


def default_ref_db_path() -> Path:
    return PROJECT_ROOT / "data" / "ref_db.h5"


def load_eval_settings(run_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    settings = {}

    # Prefer stored eval settings if you later decide to put them into run_config.json.
    if isinstance(run_cfg.get("eval_settings"), dict):
        settings.update(run_cfg["eval_settings"])

    settings.setdefault("device", args.device)
    settings.setdefault("default_dtype", args.default_dtype)
    settings.setdefault("frechet", not args.no_frechet)
    settings.setdefault("fmax", args.fmax)
    settings.setdefault("compare_crystal_modes", args.compare_crystal_modes)
    settings.setdefault("mode_skip_first", args.mode_skip_first)
    settings.setdefault("mode_degeneracy_tol", args.mode_degeneracy_tol)
    settings.setdefault("write_ref_db", True)

    settings["use_ref_db_cache"] = not args.no_db_cache
    settings["force_reeval"] = args.force_reeval
    settings["plot_only"] = args.plot_only

    return settings


def eval_complete(run_dir: Path, structure: str) -> bool:
    return (
        (run_dir / f"{structure}_eval_summary.json").exists()
        and (run_dir / f"{structure}_eval_arrays.npz").exists()
    )


def passes_filters(run_dir: Path, structure: str, args: argparse.Namespace) -> bool:
    text = str(run_dir)

    if args.pbe_only and "PBE" not in structure:
        return False

    if args.structure and structure != args.structure:
        return False

    if args.contains and args.contains not in text:
        return False

    if args.exclude and args.exclude in text:
        return False

    return True


# ============================================================
# evaluation
# ============================================================
def recompute_score_from_db(
    *,
    ref_db_path: Path,
    structure: str,
    dataset_split: str,
    sweep_id: str,
    run_id: str,
) -> dict[str, Any]:

    cached = read_model_evaluation(
        ref_db_path=ref_db_path,
        structure=structure,
        run_id=run_id,
        dataset_split=dataset_split,
        sweep_id=sweep_id,
    )

    ranking_metrics = dict(
        cached.get("ranking_metrics", {})
    )
    print(f'Old score: {ranking_metrics.get("composite_score"):5.2f}')

    composite_score, score_parts = make_composite_score(
        freq_mae_ir_cm1=ranking_metrics.get(
            "freq_mae_ir_cm1"
        ),
        freq_mae_ir_weighted_cm1=ranking_metrics.get(
            "freq_mae_ir_weighted_cm1"
        ),
        spectrum_rel_l2=ranking_metrics.get(
            "spectrum_rel_l2"
        ),
        intensity_pearson_r=ranking_metrics.get(
            "intensity_pearson_r"
        ),
        mean_mode_overlap=ranking_metrics.get(
            "crystal_mode_mean_overlap"
        ),
        mean_subspace_overlap=ranking_metrics.get(
            "crystal_mode_mean_subspace_overlap"
        ),
    )

    print(f'New score: {composite_score:5.2f}')
    ranking_metrics["composite_score"] = composite_score

    for key, value in score_parts.items():
        ranking_metrics[f"score_{key}"] = value

    update_model_ranking_metrics(
        ref_db_path=ref_db_path,
        structure=structure,
        run_id=run_id,
        dataset_split=dataset_split,
        sweep_id=sweep_id,
        ranking_metrics=ranking_metrics,
    )

    return ranking_metrics


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_cfg_path = run_dir / "run_config.json"
    run_cfg = load_json(run_cfg_path)

    run_name = str(run_cfg.get("run_name", run_dir.name))
    structure = infer_structure(run_dir, run_cfg)
    dataset_split = infer_dataset_split(run_dir)
    sweep_id = infer_sweep_id(run_dir)

    model_path = latest_model_path(run_dir)
    cif_path = Path(args.cif_path).resolve() if args.cif_path else default_cif_path(structure)
    ref_db_path = Path(args.ref_db).resolve() if args.ref_db else default_ref_db_path()
    eval_settings = load_eval_settings(run_cfg, args)

    train_result = {
        "run_name": run_name,
        "status": "ok",
        "result_dir": str(run_dir),
        "run_config_path": str(run_cfg_path),
        "trained_model_path": str(model_path),
        "deploy_model_path": None,
        "error": None,
    }

    if args.score_only:

        ranking_metrics = recompute_score_from_db(
            ref_db_path=ref_db_path,
            structure=structure,
            dataset_split=dataset_split,
            sweep_id=sweep_id,
            run_id=run_name,
        )

        return {
            "status": "score_updated",
            "run_name": run_name,
            "structure": structure,
            "dataset_split": dataset_split,
            "sweep_id": sweep_id,
            "run_dir": str(run_dir),
            "ranking_metrics": ranking_metrics,
        }

    if args.eval_missing and eval_complete(run_dir, structure):
        return {
            "status": "skipped_complete",
            "run_name": run_name,
            "structure": structure,
            "run_dir": str(run_dir),
            "model_path": str(model_path),
        }

    if args.dry_run:
        return {
            "status": "dry_run",
            "run_name": run_name,
            "structure": structure,
            "dataset_split": dataset_split,
            "sweep_id": sweep_id,
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "cif_path": str(cif_path),
            "ref_db_path": str(ref_db_path),
        }

    summary = evaluate_model(
        model_path=model_path,
        structure=structure,
        cif_path=cif_path,
        output_dir=run_dir,
        crystal_db_path=ref_db_path,
        device=eval_settings["device"],
        default_dtype=eval_settings["default_dtype"],
        frechet=eval_settings["frechet"],
        fmax=eval_settings["fmax"],
        compare_crystal_modes=eval_settings["compare_crystal_modes"],
        mode_skip_first=eval_settings["mode_skip_first"],
        mode_degeneracy_tol=eval_settings["mode_degeneracy_tol"],
        write_ref_db=eval_settings["write_ref_db"],
        ref_db_path=ref_db_path,
        run_id=run_name,
        dataset_split=dataset_split,
        sweep_id=sweep_id,
        use_ref_db_cache=eval_settings["use_ref_db_cache"],
        force_reeval=eval_settings["force_reeval"],
        plot_only=eval_settings["plot_only"],
    )

    return {
        "status": "ok",
        "run_name": run_name,
        "structure": structure,
        "dataset_split": dataset_split,
        "sweep_id": sweep_id,
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "summary": summary,
        "train_result": train_result,
    }


def update_sweep_summary(result: dict[str, Any]) -> None:
    if result["status"] != "ok":
        return

    run_dir = Path(result["run_dir"])
    sweep_root = run_dir.parent

    master_summary_path = sweep_root / "master_summary.json"
    master_csv_path = sweep_root / "master_summary.csv"
    best_model_path = sweep_root / "best_model.json"

    if master_summary_path.exists():
        master_summary = load_json(master_summary_path)
    else:
        master_summary = {
            "sweep_name": sweep_root.name,
            "evaluation_results": [],
            "training_results": [],
            "best_model": None,
        }

    run_name = result["run_name"]
    train_result = result["train_result"]
    eval_result = result["summary"]

    record = {
        "run_name": run_name,
        "status": "ok",
        "result_dir": str(run_dir),
        "metrics": eval_result.get("ranking_metrics", {}),
        "artifacts": eval_result.get("artifacts", {}),
        "ref_db": eval_result.get("ref_db", {}),
    }

    old = master_summary.get("evaluation_results", [])
    old = [r for r in old if r.get("run_name") != run_name]
    old.append(record)
    master_summary["evaluation_results"] = old

    row = build_eval_row(
        train_result=train_result,
        eval_result=eval_result,
        error=None,
    )
    append_master_csv(row, master_csv_path)

    master_summary["best_model"] = choose_best_model(master_summary)
    save_json(master_summary, master_summary_path)

    if master_summary["best_model"] is not None:
        save_json(master_summary["best_model"], best_model_path)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively rerun plotting/evaluation for all trained models in results/."
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT_DEFAULT,
        help="Root to scan. Default: PROJECT_ROOT/results",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Only rebuild plots/summaries from ref_db cache.",
    )
    mode.add_argument(
        "--eval-missing",
        action="store_true",
        help="Evaluate only runs without local *_eval_summary.json and *_eval_arrays.npz.",
    )
    mode.add_argument(
        "--force-reeval",
        action="store_true",
        help="Ignore DB cache and recompute MACE inference.",
    )

    parser.add_argument(
        "--pbe-only",
        action="store_true",
        help="Only evaluate runs whose inferred structure contains 'PBE'.",
    )

    mode.add_argument(
        "--score-only",
        action="store_true",
        help="Only recompute composite score from cached DB metrics and write ranking_metrics back to DB.",
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-db-cache", action="store_true")

    parser.add_argument("--structure", default=None)
    parser.add_argument("--contains", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--max-runs", type=int, default=None)

    parser.add_argument("--ref-db", default=None)
    parser.add_argument("--cif-path", default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--no-frechet", action="store_true")
    parser.add_argument("--fmax", type=float, default=1e-11)

    parser.add_argument("--compare-crystal-modes", action="store_true")
    parser.add_argument("--mode-skip-first", type=int, default=3)
    parser.add_argument("--mode-degeneracy-tol", type=float, default=1.0)

    parser.add_argument(
        "--no-update-master-summary",
        action="store_true",
        help="Do not update per-sweep master_summary.json/master_summary.csv.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_conda_env()

    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Missing results root: {results_root}")

    discovered = discover_run_dirs(results_root)

    selected = []
    for run_dir in discovered:
        try:
            run_cfg = load_json(run_dir / "run_config.json")
            structure = infer_structure(run_dir, run_cfg)
            if passes_filters(run_dir, structure, args):
                selected.append(run_dir)
        except Exception:
            continue

    if args.max_runs is not None:
        selected = selected[:args.max_runs]

    batch_log = {
        "results_root": str(results_root),
        "n_discovered": len(discovered),
        "n_selected": len(selected),
        "runs": [],
    }

    print(f"Discovered trained runs : {len(discovered)}")
    print(f"Selected runs           : {len(selected)}")

    for i, run_dir in enumerate(selected, start=1):
        print("\n" + "=" * 90)
        print(f"[{i:04d}/{len(selected):04d}] {run_dir}")
        print("=" * 90)

        try:
            result = evaluate_run(run_dir, args)

            if (
                result["status"] == "ok"
                and not args.dry_run
                and not args.no_update_master_summary
            ):
                update_sweep_summary(result)

            compact = {
                k: v for k, v in result.items()
                if k not in {"summary", "train_result"}
            }
            batch_log["runs"].append(compact)
            print(f"Status: {result['status']}")

        except Exception as exc:
            fail = {
                "status": "failed",
                "run_dir": str(run_dir),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            batch_log["runs"].append(fail)
            print("Status: failed")
            print(repr(exc))

    log_path = results_root / "rerun_results_summary.json"
    save_json(batch_log, log_path)

    print("\nFinished.")
    print(f"Batch log: {log_path}")


if __name__ == "__main__":
    main()

