# util/eval.py

from __future__ import annotations

import csv
import math
import re
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from util.io import ensure_dir, load_json, save_json


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_mace_rmse_table_from_train_log(train_log_path: Path) -> dict[str, Any]:
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

    for config_type, rmse_e, rmse_f, rel_f in row_pattern.findall(text):
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


def choose_model_path(
    train_result: dict[str, Any],
    use_deploy_model_if_available: bool = False,
) -> Path:
    deploy_model_path = train_result.get("deploy_model_path")
    trained_model_path = train_result.get("trained_model_path")

    if use_deploy_model_if_available and deploy_model_path:
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
    keys = [
        "seed",
        "r_max",
        "batch_size",
        "valid_batch_size",
        "energy_weight",
        "forces_weight",
        "use_stress",
        "stress_weight",
        "max_epochs",
    ]

    if run_cfg is None:
        return {k: None for k in keys}

    return {k: run_cfg.get(k) for k in keys}


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
    use_deploy_model_if_available: bool = False,
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
        **hp,
        **train_log_metrics,
        "n_imag_modes": None,
        "n_ir_active_modes": None,
        "zpe_eV": None,
        "spectrum_rel_l2": None,
        **rm,
        "summary_json": None,
        "ir_plot": None,
        "error": error,
    }

    if train_result.get("status") == "ok":
        try:
            row["model_path"] = str(
                choose_model_path(
                    train_result,
                    use_deploy_model_if_available=use_deploy_model_if_available,
                )
            )
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


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def build_best_model_record(
    eval_record: dict[str, Any],
    use_deploy_model_if_available: bool = False,
) -> dict[str, Any]:
    row = build_eval_row(
        eval_record["train_result"],
        eval_record["eval_result"],
        error=None,
        use_deploy_model_if_available=use_deploy_model_if_available,
    )

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


def choose_best_model(
    master_summary: dict[str, Any],
    use_deploy_model_if_available: bool = False,
) -> dict[str, Any] | None:
    ok_records = [
        rec for rec in master_summary.get("evaluation_results", [])
        if rec.get("status") == "ok" and "eval_result" in rec
    ]

    if not ok_records:
        return None

    best = min(ok_records, key=evaluation_sort_key)
    return build_best_model_record(
        best,
        use_deploy_model_if_available=use_deploy_model_if_available,
    )


def eval_cif_structure_name(structure: str) -> str:
    for suffix in ("_PBE", "_PBESOLXC", "_PBESOL", "_HSESOL", "_HSE"):
        if structure.upper().endswith(suffix):
            return structure[: -len(suffix)]
    return structure


def find_existing_run_dirs(sweep_root: Path) -> list[Path]:
    run_dirs: list[Path] = []

    for child in sorted(sweep_root.iterdir()):
        if not child.is_dir():
            continue

        if not (child / "run_config.json").exists():
            continue

        if is_training_complete(child):
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


def is_training_complete(run_dir: Path) -> bool:
    models_dir = run_dir / "models"
    return models_dir.exists() and any(models_dir.glob("*.model"))


def is_eval_complete(run_dir: Path, structure: str) -> bool:
    return (
        (run_dir / f"{structure}_eval_summary.json").exists()
        and (run_dir / f"{structure}_eval_arrays.npz").exists()
    )


def make_compact_eval_record(
    run_name: str,
    train_result: dict[str, Any],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    return {
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


def run_single_model_eval(
    *,
    evaluate_model_func,
    train_result: dict[str, Any],
    structure: str,
    cif_path: Path,
    crystal_db_path: Path,
    eval_settings: dict[str, Any],
    dataset_split: str,
    sweep_id: str,
    use_deploy_model_if_available: bool = False,
) -> dict[str, Any]:
    run_name = train_result.get("run_name", "<unknown>")
    model_path = choose_model_path(
        train_result,
        use_deploy_model_if_available=use_deploy_model_if_available,
    )
    result_dir = Path(train_result["result_dir"])

    # print(eval_settings)
    return evaluate_model_func(
        model_path=model_path,
        structure=structure,
        cif_path=cif_path,
        output_dir=result_dir,
        crystal_db_path=crystal_db_path,
        device=eval_settings["device"],
        default_dtype=eval_settings["default_dtype"],
        frechet=eval_settings["frechet"],
        fmax=eval_settings["fmax"],
        compare_crystal_modes=eval_settings["compare_crystal_modes"],
        crystal_hess_path=eval_settings["crystal_hess_path"],
        crystal_freq_out_path=eval_settings["crystal_freq_out_path"],
        crystal_hessian_units=eval_settings["crystal_hessian_units"],
        run_phonopy=eval_settings.get("run_phonopy", False),
        phonopy_plugin=eval_settings.get("phonopy_plugin"),
        write_ref_db=eval_settings.get("write_ref_db", False),
        ref_db_path=eval_settings.get("ref_db_path"),
        run_id=run_name,
        dataset_split=dataset_split,
        sweep_id=sweep_id,
    )


def preview_existing_eval_runs(
    *,
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

            previews.append({
                "run_name": run_cfg.get("run_name", run_dir.name),
                "run_dir": str(run_dir),
                "run_config": str(run_config_path),
                "model_path": str(model_path),
                "eval_complete": is_eval_complete(run_dir, structure),
                "eval_summary_exists": (run_dir / f"{structure}_eval_summary.json").exists(),
                "eval_arrays_exists": (run_dir / f"{structure}_eval_arrays.npz").exists(),
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
    *,
    evaluate_model_func,
    project_root: Path,
    sweep_root: Path,
    structure: str,
    cif_path: Path,
    crystal_db_path: Path,
    eval_settings: dict[str, Any],
    max_eval_runs: int | None = None,
    dry_run: bool = False,
    skip_completed: bool = True,
    overwrite_summary: bool = False,
    use_deploy_model_if_available: bool = False,
) -> None:
    if not cif_path.exists():
        raise FileNotFoundError(f"Missing CIF: {cif_path}")

    if not crystal_db_path.exists():
        raise FileNotFoundError(f"Missing CRYSTAL DB: {crystal_db_path}")

    run_dirs = find_existing_run_dirs(sweep_root)

    if max_eval_runs is not None:
        run_dirs = run_dirs[:max_eval_runs]

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
            print(f"  eval_complete: {preview.get('eval_complete')}")
            print(f"  eval_summary_exists: {preview.get('eval_summary_exists')}")
            print(f"  eval_arrays_exists: {preview.get('eval_arrays_exists')}")
            if preview.get("error"):
                print(f"  error: {preview['error']}")
            print()

        return

    master_summary_path = sweep_root / "master_summary.json"
    master_csv_path = sweep_root / "master_summary.csv"
    best_model_path = sweep_root / "best_model.json"

    if overwrite_summary and master_csv_path.exists():
        master_csv_path.unlink()

    if overwrite_summary or not master_summary_path.exists():
        master_summary: dict[str, Any] = {
            "sweep_name": sweep_root.name,
            "created_at": now_iso(),
            "mode": "eval_only",
            "project_root": str(project_root),
            "sweep_root": str(sweep_root),
            "structure": structure,
            "cif_path": str(cif_path),
            "crystal_db_path": str(crystal_db_path),
            "max_eval_runs": max_eval_runs,
            "use_deploy_model_if_available": use_deploy_model_if_available,
            "eval_settings": deepcopy(eval_settings),
            "n_existing_runs_found": len(run_dirs),
            "training_results": [],
            "evaluation_results": [],
            "best_model": None,
        }
    else:
        master_summary = load_json(master_summary_path)
        master_summary.setdefault("training_results", [])
        master_summary.setdefault("evaluation_results", [])

    save_json(master_summary, master_summary_path)

    print(f"Eval-only sweep root: {sweep_root}")
    print(f"Run directories found: {len(run_dirs)}")

    for run_dir in run_dirs:
        if skip_completed and is_eval_complete(run_dir, structure):
            print(f"[SKIP] Evaluation already complete: {run_dir.name}")
            continue

        train_result = build_train_result_from_existing_run(run_dir)
        master_summary["training_results"].append(train_result)
        save_json(master_summary, master_summary_path)

        run_name = train_result.get("run_name", run_dir.name)

        try:
            print(f"\n=== Starting evaluation for existing run: {run_name} ===")

            eval_result = run_single_model_eval(
                evaluate_model_func=evaluate_model_func,
                train_result=train_result,
                structure=structure,
                cif_path=cif_path,
                crystal_db_path=crystal_db_path,
                eval_settings=eval_settings,
                dataset_split=sweep_root.parent.name,
                sweep_id=sweep_root.name,
                use_deploy_model_if_available=use_deploy_model_if_available,
            )

            eval_record = make_compact_eval_record(run_name, train_result, eval_result)
            master_summary["evaluation_results"].append(eval_record)

            row = build_eval_row(
                train_result=train_result,
                eval_result=eval_result,
                error=None,
                use_deploy_model_if_available=use_deploy_model_if_available,
            )
            append_master_csv(row, master_csv_path)

            best_model = choose_best_model(
                master_summary,
                use_deploy_model_if_available=use_deploy_model_if_available,
            )
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

            row = build_eval_row(
                train_result=train_result,
                eval_result=None,
                error=repr(exc),
                use_deploy_model_if_available=use_deploy_model_if_available,
            )
            append_master_csv(row, master_csv_path)

            best_model = choose_best_model(
                master_summary,
                use_deploy_model_if_available=use_deploy_model_if_available,
            )
            master_summary["best_model"] = best_model
            save_json(master_summary, master_summary_path)

            if best_model is not None:
                save_json(best_model, best_model_path)

            print(f"Evaluation failed for: {run_name}")
            print(repr(exc))

    master_summary["best_model"] = choose_best_model(
        master_summary,
        use_deploy_model_if_available=use_deploy_model_if_available,
    )
    master_summary["finished_at"] = now_iso()
    save_json(master_summary, master_summary_path)

    if master_summary["best_model"] is not None:
        save_json(master_summary["best_model"], best_model_path)

    print("\nEval-only sweep finished.")
    print(f"Master summary JSON: {master_summary_path}")
    print(f"Master summary CSV : {master_csv_path}")
    print(f"Best model JSON    : {best_model_path}")
    