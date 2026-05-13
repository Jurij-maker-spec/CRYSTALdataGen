#!/usr/bin/env python3
"""
Refactored inference / evaluation script for CRYSTALdataGen.

Designed to be:
1. imported and called from run_master_eval.py
2. runnable directly from the command line

What it does
------------
- read CIF
- convert to primitive cell
- run geometry optimization
- extract LES/BEC outputs
- compute phonon frequencies from analytical Hessian
- compute relative IR intensities
- compare broadened IR spectrum to CRYSTAL reference
- save plots and JSON summaries into a chosen output directory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import spglib

from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from ase import Atoms
from ase.io import read
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
import matplotlib as mpl
mpl.rcParams['font.size'] = 12

# ------------------------------------------------------------------
# project imports
# ------------------------------------------------------------------

PYTHON_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_SCRIPTS_ROOT))

from util.plotting import plot_ir_spectrum as ir_spec_plotter
from util.ir import (
    asr_correct_bec,
    mass_unweight_eigenvectors,
    normalize_modes_cartesian,
    mode_effective_charges,
    broaden_spectrum,
    print_ir_modes,
)
from util.phonons import (
    freqs_from_analytical_hessian,
)
from util.mode_matching import (
    run_mode_comparison,
    run_mode_comparison_from_ref_db,
    print_mode_match_summary,
    print_group_match_summary,
)
from util.ref_db import write_model_evaluation

# ------------------------------------------------------------------
# defaults
# ------------------------------------------------------------------

# DEFAULT_ROOT = Path("/home/jha/jha/python_scripts/CRYSTALdataGen")
DEFAULT_ROOT = Path(__file__).resolve().parent

DEFAULT_CIF_ROOT = DEFAULT_ROOT / "inference" / "CIFs"
DEFAULT_CRYSTAL_DB = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5")
DEFAULT_CRYSTAL_STRUCTURES_ROOT = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/structures")
DEFAULT_CRYSTAL_HESSIAN_UNITS = "hartree/bohr^2"


# ------------------------------------------------------------------
# utilities
# ------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_serializable(obj):
    """Convert numpy / Path objects to JSON-serializable objects."""
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


def get_primitive_atoms_from_cif(cif_path: Path) -> Atoms:
    atoms = read(cif_path, format="cif")
    atoms.set_pbc(True)

    cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    primitive = spglib.standardize_cell(cell, to_primitive=True, no_idealize=False)
    if primitive is None:
        raise RuntimeError(f"spglib.standardize_cell returned None for {cif_path}")

    prim_atoms = Atoms(
        numbers=primitive[2],
        scaled_positions=primitive[1],
        cell=primitive[0],
        pbc=True,
    )
    return prim_atoms


def calculator(model_path: Path, device: str = "cuda", default_dtype: str = "float64", cal: str = "0"):
    if cal == "0":
        from mace.calculators import MACECalculator

        calc = MACECalculator(
            model_paths=str(model_path),
            default_dtype=default_dtype,
            device=device,
        )
        return calc
    else:
        raise ValueError(f"Unknown calculator mode: {cal}")


def geometry_optimisation(atoms: Atoms, frechet: bool = True, fmax: float = 1e-11, trajectory: Path | None = None) -> Atoms:
    ei = atoms.get_potential_energy()
    print("Initial Energy:", ei, "eV")

    traj_path = str(trajectory) if trajectory is not None else None

    if frechet:
        atoms.set_constraint(FixSymmetry(atoms))
        ecf = FrechetCellFilter(atoms)
        opt = LBFGS(ecf, trajectory=traj_path)
    else:
        opt = LBFGS(atoms, trajectory=traj_path)

    opt.run(fmax=fmax)

    ef = atoms.get_potential_energy()
    print("Final Energy:", ef, "eV")
    return atoms


def get_les_outputs(calc, atoms: Atoms, compute_forces: bool = False, compute_stress: bool = False):
    """
    Directly call the underlying MACELES model and request LES outputs.
    """
    batch = calc._atoms_to_batch(atoms)
    model = calc.models[0]

    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    for key in batch.keys:
        val = batch[key]
        if torch.is_tensor(val):
            val = val.to(model_device)
            if torch.is_floating_point(val):
                val = val.to(dtype=model_dtype)
            batch[key] = val

    model.eval()
    out = model(
        batch.to_dict(),
        training=False,
        compute_force=compute_forces,
        compute_stress=compute_stress,
        compute_bec=True,
    )

    out_np = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            out_np[k] = v.detach().cpu().numpy()
        else:
            out_np[k] = v
    return out_np


def summarize_bec(bec: np.ndarray | None) -> dict:
    """
    Keep BEC computation for IR intensities, but no explicit reference comparison.
    """
    if bec is None:
        return {
            "bec_present": False,
            "bec_shape": None,
            "asr_sum_before": None,
            "asr_sum_after": None,
        }

    bec = np.asarray(bec, dtype=float)
    if bec.ndim != 3 or bec.shape[1:] != (3, 3):
        return {
            "bec_present": True,
            "bec_shape": list(bec.shape),
            "asr_sum_before": None,
            "asr_sum_after": None,
        }

    bec_corr = asr_correct_bec(bec)

    return {
        "bec_present": True,
        "bec_shape": list(bec.shape),
        "asr_sum_before": bec.sum(axis=0),
        "asr_sum_after": bec_corr.sum(axis=0),
    }


def spectrum_distance_l2(x_ref, y_ref, x_pred, y_pred):
    """
    Compare two spectra on a common grid.
    """
    x_min = max(float(np.min(x_ref)), float(np.min(x_pred)))
    x_max = min(float(np.max(x_ref)), float(np.max(x_pred)))

    if x_max <= x_min:
        return None

    x_common = np.linspace(x_min, x_max, 4000)
    y_ref_i = np.interp(x_common, x_ref, y_ref)
    y_pred_i = np.interp(x_common, x_pred, y_pred)

    denom = np.linalg.norm(y_ref_i)
    if denom <= 0.0:
        return None

    rel_l2 = np.linalg.norm(y_pred_i - y_ref_i) / denom
    return float(rel_l2)


def normalize_nonnegative(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = np.sum(x)
    if s > 0.0:
        return x / s
    return x


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    try:
        r, _ = pearsonr(x, y)
        return float(r)
    except Exception:
        return None


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    try:
        r, _ = spearmanr(x, y)
        return float(r)
    except Exception:
        return None


def match_ir_modes(
    pred_freqs_cm: np.ndarray,
    pred_intensities: np.ndarray,
    ref_freqs_cm: np.ndarray,
    ref_intensities: np.ndarray,
) -> dict:
    """
    Match predicted and reference IR-active modes by minimizing absolute
    frequency differences with Hungarian assignment.

    Returns a dict with matched arrays and scalar metrics.
    """
    pred_freqs_cm = np.asarray(pred_freqs_cm, dtype=float)
    pred_intensities = np.asarray(pred_intensities, dtype=float)
    ref_freqs_cm = np.asarray(ref_freqs_cm, dtype=float)
    ref_intensities = np.asarray(ref_intensities, dtype=float)

    out = {
        "matched_mode_count": 0,
        "pred_freqs_matched_cm": np.array([], dtype=float),
        "ref_freqs_matched_cm": np.array([], dtype=float),
        "pred_intensities_matched": np.array([], dtype=float),
        "ref_intensities_matched": np.array([], dtype=float),
        "freq_abs_errors_cm": np.array([], dtype=float),
        "freq_mae_ir_cm1": None,
        "freq_rmse_ir_cm1": None,
        "freq_mae_ir_weighted_cm1": None,
        "intensity_pearson_r": None,
        "intensity_spearman_r": None,
    }

    if len(pred_freqs_cm) == 0 or len(ref_freqs_cm) == 0:
        return out

    pred_i_norm = normalize_nonnegative(pred_intensities)
    ref_i_norm = normalize_nonnegative(ref_intensities)

    cost = np.abs(pred_freqs_cm[:, None] - ref_freqs_cm[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)

    pred_f_m = pred_freqs_cm[row_ind]
    ref_f_m = ref_freqs_cm[col_ind]
    pred_i_m = pred_i_norm[row_ind]
    ref_i_m = ref_i_norm[col_ind]

    abs_err = np.abs(pred_f_m - ref_f_m)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred_f_m - ref_f_m) ** 2)))

    # Use normalized reference intensities as weights.
    w = ref_i_m.copy()
    if np.sum(w) > 0.0:
        w = w / np.sum(w)
        wmae = float(np.sum(w * abs_err))
    else:
        wmae = mae

    out.update({
        "matched_mode_count": int(len(row_ind)),
        "pred_freqs_matched_cm": pred_f_m,
        "ref_freqs_matched_cm": ref_f_m,
        "pred_intensities_matched": pred_i_m,
        "ref_intensities_matched": ref_i_m,
        "freq_abs_errors_cm": abs_err,
        "freq_mae_ir_cm1": mae,
        "freq_rmse_ir_cm1": rmse,
        "freq_mae_ir_weighted_cm1": wmae,
        "intensity_pearson_r": safe_corrcoef(pred_i_m, ref_i_m),
        "intensity_spearman_r": safe_spearman(pred_i_m, ref_i_m),
    })

    return out


def make_composite_score(
    n_imag_modes: int,
    freq_mae_ir_cm1: float | None,
    freq_mae_ir_weighted_cm1: float | None,
    spectrum_rel_l2: float | None,
    intensity_pearson_r: float | None,
) -> float | None:
    """
    Lower is better.
    To make the evaluation more sensitive to correct peak positions
    1. stronger penalty for frequency deviations
    2. non-linear amplification of large frequency errors
    3. reduced dominance of spectrum L2
    => now more frequency responsive
    """
    if freq_mae_ir_cm1 is None and spectrum_rel_l2 is None:
        return None

    score = 0.0
    score += 10.0 * float(n_imag_modes)

    if freq_mae_ir_cm1 is not None:
        score += 3.0 * float(freq_mae_ir_cm1)

    if freq_mae_ir_weighted_cm1 is not None:
        score += 2.0 * float(freq_mae_ir_weighted_cm1)

    if spectrum_rel_l2 is not None:
        score += 20.0 * float(spectrum_rel_l2)

    if intensity_pearson_r is not None:
        score += 10.0 * (1.0 - float(intensity_pearson_r))
    else:
        score += 10.0

    return float(score)


def plot_ir_spectrum(
    freqs_cm,
    intensities,
    nu_grid,
    ir_spec,
    structure: str,
    crystal_db_path: Path,
    outfile: Path,
):
    """
    Save comparison plot wrapper and return numerical CRYSTAL comparison data if available.
    """
    crystal_summary = {
        "has_crystal_reference": False,
        "crystal_freqs_cm": None,
        "crystal_intensities_rel": None,
        "crystal_kde_x": None,
        "crystal_kde_y": None,
        "spectrum_rel_l2": None,
    }
    try:
        f_crys, I_crys_rel, x_ref, kde_ref = ir_spec_plotter(freqs_cm=freqs_cm, 
                                                             intensities=intensities, 
                                                             nu_grid=nu_grid, 
                                                             ir_spec=ir_spec, 
                                                             structure=structure, 
                                                             crystal_db_path=crystal_db_path,
                                                             outfile=outfile)

        crystal_summary["has_crystal_reference"] = True
        crystal_summary["crystal_freqs_cm"] = np.asarray(f_crys, dtype=float)
        crystal_summary["crystal_intensities_rel"] = np.asarray(I_crys_rel, dtype=float)
        crystal_summary["crystal_kde_x"] = np.asarray(x_ref, dtype=float)
        crystal_summary["crystal_kde_y"] = np.asarray(kde_ref, dtype=float)
        crystal_summary["spectrum_rel_l2"] = spectrum_distance_l2(
            x_ref, kde_ref, nu_grid, ir_spec
        )
    except Exception as exc:
        print(f"----> Writing summary for {structure} failed: {exc}")

    return crystal_summary


def find_ir_active_modes(freqs_cm, imag_flags, intensities, threshold=1e-8):
    freqs_cm = np.asarray(freqs_cm, dtype=float)
    imag_flags = np.asarray(imag_flags, dtype=bool)
    intensities = np.asarray(intensities, dtype=float)

    mask = (~imag_flags) & (freqs_cm > 1e-6) & (intensities > threshold)
    return freqs_cm[mask], intensities[mask], mask


def default_crystal_hessian_path(structure: str, crystal_structures_root: str | Path = DEFAULT_CRYSTAL_STRUCTURES_ROOT) -> Path:
    """Default CRYSTAL hessfreq path used by the explicit mode-overlap workflow."""
    return Path(crystal_structures_root).resolve() / structure / "freq" / f"{structure}_freq.hessfreq"


def default_crystal_freq_out_path(structure: str, crystal_structures_root: str | Path = DEFAULT_CRYSTAL_STRUCTURES_ROOT) -> Path:
    """Default CRYSTAL freq.out path used by the explicit mode-overlap workflow."""
    return Path(crystal_structures_root).resolve() / structure / "freq" / f"{structure}_freq.out"


def empty_crystal_mode_comparison_summary() -> dict:
    return {
        "enabled": False,
        "success": False,
        "error": None,
        "crystal_hess_path": None,
        "freq_out_path": None,
        "crystal_hessian_units": None,
        "skip_first": None,
        "degeneracy_tol": None,
        "heatmap_outfile": None,
        "group_heatmap_outfile": None,
        "n_matches": 0,
        "n_subgroups": 0,
        "mean_abs_freq_error_cm1": None,
        "max_abs_freq_error_cm1": None,
        "mean_mode_overlap": None,
        "min_mode_overlap": None,
        "mean_subspace_overlap": None,
        "min_subspace_overlap": None,
        "matches": [],
        "subgroups": [],
    }


def _as_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_crystal_mode_comparison(comparison: dict, crystal_hess_path: Path, freq_out_path: Path | None, crystal_hessian_units: str, skip_first: int, degeneracy_tol: float, heatmap_outfile: Path) -> dict:
    """Condense the verbose ModeComparison output for JSON summaries and ranking CSVs."""
    out = empty_crystal_mode_comparison_summary()
    out.update({
        "enabled": True,
        "success": True,
        "crystal_hess_path": crystal_hess_path,
        "freq_out_path": freq_out_path,
        "crystal_hessian_units": crystal_hessian_units,
        "skip_first": int(skip_first),
        "degeneracy_tol": float(degeneracy_tol),
        "heatmap_outfile": heatmap_outfile,
        "group_heatmap_outfile": comparison.get("group_heatmap_outfile"),
    })
    matches = comparison.get("matches", []) or []
    subgroups = comparison.get("subgroups", []) or []
    out["n_matches"] = int(len(matches))
    out["n_subgroups"] = int(len(subgroups))
    out["matches"] = matches
    out["subgroups"] = subgroups
    freq_errors, overlaps = [], []
    for m in matches:
        if not isinstance(m, dict):
            continue
        err = None
        for key in ("abs_freq_error", "abs_freq_error_cm1", "freq_abs_error", "freq_abs_error_cm1"):
            if key in m:
                err = _as_float_or_none(m.get(key))
                break
        if err is None:
            rf = _as_float_or_none(m.get("ref_freq_cm") or m.get("freq_ref_cm") or m.get("ref_freq"))
            tf = _as_float_or_none(m.get("test_freq_cm") or m.get("freq_test_cm") or m.get("test_freq"))
            if rf is not None and tf is not None:
                err = abs(tf - rf)
        if err is not None:
            freq_errors.append(err)
        ov = None
        for key in ("overlap", "mode_overlap", "abs_overlap"):
            if key in m:
                ov = _as_float_or_none(m.get(key))
                break
        if ov is not None:
            overlaps.append(abs(ov))
    subspace_overlaps = []
    for g in subgroups:
        if isinstance(g, dict) and "subspace_overlap" in g:
            ov = _as_float_or_none(g.get("subspace_overlap"))
            if ov is not None:
                subspace_overlaps.append(ov)
    if freq_errors:
        out["mean_abs_freq_error_cm1"] = float(np.mean(freq_errors))
        out["max_abs_freq_error_cm1"] = float(np.max(freq_errors))
    if overlaps:
        out["mean_mode_overlap"] = float(np.mean(overlaps))
        out["min_mode_overlap"] = float(np.min(overlaps))
    if subspace_overlaps:
        out["mean_subspace_overlap"] = float(np.mean(subspace_overlaps))
        out["min_subspace_overlap"] = float(np.min(subspace_overlaps))
    return out


def run_optional_crystal_mode_comparison(
        atoms: Atoms, 
        mace_modes: dict, 
        structure: str, 
        output_dir: Path, 
        enabled: bool = False, 
        crystal_hess_path: str | Path | None = None, 
        freq_out_path: str | Path | None = None, 
        crystal_structures_root: str | Path = DEFAULT_CRYSTAL_STRUCTURES_ROOT, 
        crystal_hessian_units: str = DEFAULT_CRYSTAL_HESSIAN_UNITS, 
        skip_first: int = 3, 
        degeneracy_tol: float = 1.0,
        ref_db_path: str | Path | None = None,
        use_ref_db: bool = True,
    ) -> dict:
    """
    Optional explicit mode matching against CRYSTAL hessfreq data.

    This ports the single_inference.py workflow into the importable evaluator.
    It compares full Hessian eigenvectors/subspaces, saves the overlap heatmap,
    and returns a compact JSON-safe summary. It is separate from the default
    IR-peak Hungarian matching against CRYSTALreference.h5.
    """
    summary = empty_crystal_mode_comparison_summary()
    summary.update({
        "enabled": bool(enabled),
        "crystal_hessian_units": crystal_hessian_units,
        "skip_first": int(skip_first),
        "degeneracy_tol": float(degeneracy_tol),
    })
    if not enabled:
        return summary
    
    heatmap_outfile = output_dir / f"{structure}_mode_overlap.png"
    if use_ref_db and ref_db_path is not None:
        summary.update({
            "crystal_hess_path": None,
            "freq_out_path": None,
            "heatmap_outfile": heatmap_outfile,
            "source": "ref_db",
            "ref_db_path": str(ref_db_path),
        })

        try:
            print("\n=== CRYSTAL DB mode comparison ===")
            comparison = run_mode_comparison_from_ref_db(
                ref_db_path=ref_db_path,
                structure=structure,
                mace_modes=mace_modes,
                skip_first=skip_first,
                degeneracy_tol=degeneracy_tol,
                heatmap_outfile=heatmap_outfile,
                title=f"{structure}: CRYSTAL DB vs MACE mode overlap",
            )

            print_mode_match_summary(comparison.get("matches", []))
            print_group_match_summary(comparison.get("subgroups", []))

            return summarize_crystal_mode_comparison(
                comparison=comparison,
                crystal_hess_path=Path("ref_db"),
                freq_out_path=None,
                crystal_hessian_units="stored_eigvecs_mw",
                skip_first=skip_first,
                degeneracy_tol=degeneracy_tol,
                heatmap_outfile=heatmap_outfile,
            )

        except Exception as exc:
            print("\nMode comparison from ref_db failed; falling back to file parsing.")
            print(exc)
            summary["error"] = f"ref_db failed: {exc}"

    if crystal_hess_path is None:
        crystal_hess_path = default_crystal_hessian_path(structure, crystal_structures_root)
    crystal_hess_path = Path(crystal_hess_path).resolve()
    if freq_out_path is None:
        default_freq_out = default_crystal_freq_out_path(structure, crystal_structures_root)
        freq_out_path = default_freq_out if default_freq_out.exists() else None
    else:
        freq_out_path = Path(freq_out_path).resolve()
    summary.update({
        "crystal_hess_path": crystal_hess_path,
        "freq_out_path": freq_out_path,
        "heatmap_outfile": heatmap_outfile,
    })
    if not crystal_hess_path.exists():
        summary["error"] = f"No CRYSTAL Hessian found at: {crystal_hess_path}"
        print(f"\n{summary['error']}")
        return summary
    try:
        print("\n=== CRYSTAL Hessian mode comparison ===")
        comparison = run_mode_comparison(
            atoms=atoms,
            mace_modes=mace_modes,
            crystal_hess_path=crystal_hess_path,
            freq_out_path=freq_out_path,
            crystal_hessian_units=crystal_hessian_units,
            skip_first=skip_first,
            degeneracy_tol=degeneracy_tol,
            heatmap_outfile=heatmap_outfile,
            title=f"{structure}: CRYSTAL vs MACE mode overlap",
        )
        print_mode_match_summary(comparison.get("matches", []))
        print_group_match_summary(comparison.get("subgroups", []))
        group_heatmap = comparison.get("group_heatmap_outfile")
        if group_heatmap is not None:
            print(f"Saved degenerate-group heatmap to: {group_heatmap}")
        print("\nDegenerate/subspace comparison:")
        print(f"{'g_ref':>6s} {'g_test':>6s} {'ref_modes':>18s} {'test_modes':>18s} {'subspace_overlap':>18s}")
        print("-" * 70)
        for g in comparison.get("subgroups", []) or []:
            ref_modes = [int(x + 1) for x in g.get("ref_modes", [])]
            test_modes = [int(x + 1) for x in g.get("test_modes", [])]
            print(
                f"{int(g.get('group_ref_index', -1)) + 1:6d} "
                f"{int(g.get('group_test_index', -1)) + 1:6d} "
                f"{str(ref_modes):>18s} "
                f"{str(test_modes):>18s} "
                f"{float(g.get('subspace_overlap', np.nan)):18.6f}"
            )
        print(f"\nSaved mode-overlap heatmap to: {heatmap_outfile}")
        return summarize_crystal_mode_comparison(
            comparison=comparison,
            crystal_hess_path=crystal_hess_path,
            freq_out_path=freq_out_path,
            crystal_hessian_units=crystal_hessian_units,
            skip_first=skip_first,
            degeneracy_tol=degeneracy_tol,
            heatmap_outfile=heatmap_outfile,
        )
    except Exception as exc:
        summary["error"] = str(exc)
        print("\nMode comparison against CRYSTAL failed:")
        print(exc)
        return summary


def evaluate_model(
    model_path: str | Path,
    structure: str,
    cif_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    crystal_db_path: str | Path = DEFAULT_CRYSTAL_DB,
    device: str = "cuda",
    default_dtype: str = "float64",
    frechet: bool = True,
    fmax: float = 1e-11,
    calculator_mode: str = "0",
    compare_crystal_modes: bool = False,
    crystal_hess_path: str | Path | None = None,
    crystal_freq_out_path: str | Path | None = None,
    crystal_structures_root: str | Path = DEFAULT_CRYSTAL_STRUCTURES_ROOT,
    crystal_hessian_units: str = DEFAULT_CRYSTAL_HESSIAN_UNITS,
    mode_skip_first: int = 3,
    mode_degeneracy_tol: float = 1.0,
    # reserved plugin system
    run_phonopy=False,
    phonopy_plugin=None,
    # reference DB write-back
    write_ref_db: bool = False,
    ref_db_path: str | Path | None = None,
    run_id: str | None = None,
    dataset_split: str | None = None,
    sweep_id: str | None = None,
) -> dict:
    """
    Main callable entry point for run_master_eval.py.
    """
    model_path = Path(model_path).resolve()
    crystal_db_path = Path(crystal_db_path).resolve()

    if cif_path is None:
        cif_path = DEFAULT_CIF_ROOT / f"{structure}.cif"
    cif_path = Path(cif_path).resolve()

    if output_dir is None:
        output_dir = model_path.parent
    output_dir = Path(output_dir).resolve()
    ensure_dir(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF not found: {cif_path}")
    if not crystal_db_path.exists():
        raise FileNotFoundError(f"CRYSTAL DB not found: {crystal_db_path}")

    print("\n##                  Model eval                  ##")
    print("Structure:", structure)
    print("Model:", model_path)
    print("CIF:", cif_path)
    print("Output dir:", output_dir)

    traj_path = output_dir / f"{structure}_opt.traj"
    ir_plot_path = output_dir / f"{structure}_ir_comparison.png"
    summary_json_path = output_dir / f"{structure}_eval_summary.json"
    npz_path = output_dir / f"{structure}_eval_arrays.npz"

    atoms = get_primitive_atoms_from_cif(cif_path)
    natoms = len(atoms)

    calc = calculator(
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
        cal=calculator_mode,
    )
    atoms.calc = calc

    initial_forces = atoms.get_forces()
    initial_max_force = float(np.max(np.linalg.norm(initial_forces, axis=1)))
    # print("\nInitial forces:")
    # print(initial_forces)

    atoms = geometry_optimisation(
        atoms=atoms,
        frechet=frechet,
        fmax=fmax,
        trajectory=traj_path,
    )

    optimized_cell = atoms.cell.array.copy()
    optimized_scaled_positions = atoms.get_scaled_positions().copy()

    # LES / BEC extraction
    print("\n=== LES outputs ===")
    les_out = get_les_outputs(calc, atoms, compute_forces=False, compute_stress=False)

    if "les_energy" in les_out:
        print("LES energy:", les_out["les_energy"])
    if "latent_charges" in les_out and les_out["latent_charges"] is not None:
        print("latent_charges shape:", np.asarray(les_out["latent_charges"]).shape)

    bec = les_out.get("BEC", None)
    bec_summary = summarize_bec(bec)

    if bec is None:
        raise RuntimeError("Model evaluation failed: no BEC output returned.")

    bec = np.asarray(bec, dtype=float)
    bec_asr = asr_correct_bec(bec)

    # Hessian / phonons
    print("\n=== Hessian / frequencies ===")
    freqs_cm, eigvecs, imag_flags, zpe_eV, eigvals_SI = freqs_from_analytical_hessian(atoms, calc)
    mace_modes = {
        "freqs_cm": np.asarray(freqs_cm, dtype=float),
        "eigvecs_mw": np.asarray(eigvecs, dtype=float),
        "imag_flags": np.asarray(imag_flags, dtype=bool),
        "zpe_eV": float(zpe_eV),
        "eigvals_SI": np.asarray(eigvals_SI, dtype=float),
    }

    masses = atoms.get_masses()
    eigvecs_cart = mass_unweight_eigenvectors(eigvecs, masses)
    eigvecs_cart = normalize_modes_cartesian(eigvecs_cart)

    z_mode, intensities = mode_effective_charges(bec_asr, eigvecs_cart)

    print_ir_modes(freqs_cm, imag_flags, intensities, z_mode)

    nu_grid, ir_spec = broaden_spectrum(
        freqs_cm,
        intensities,
        imag_flags=imag_flags,
        fwhm=12.0,
    )

    crystal_summary = plot_ir_spectrum(
        freqs_cm=freqs_cm,
        intensities=intensities,
        nu_grid=nu_grid,
        ir_spec=ir_spec,
        structure=structure,
        crystal_db_path=crystal_db_path,
        outfile=ir_plot_path,
    )

    active_freqs, active_intensities, active_mask = find_ir_active_modes(
        freqs_cm=freqs_cm,
        imag_flags=imag_flags,
        intensities=intensities,
    )

    mode_match_summary = {
        "matched_mode_count": 0,
        "pred_freqs_matched_cm": None,
        "ref_freqs_matched_cm": None,
        "pred_intensities_matched": None,
        "ref_intensities_matched": None,
        "freq_abs_errors_cm": None,
        "freq_mae_ir_cm1": None,
        "freq_rmse_ir_cm1": None,
        "freq_mae_ir_weighted_cm1": None,
        "intensity_pearson_r": None,
        "intensity_spearman_r": None,
    }

    if crystal_summary.get("has_crystal_reference", False):
        ref_freqs = np.asarray(crystal_summary["crystal_freqs_cm"], dtype=float)
        ref_intensities = np.asarray(crystal_summary["crystal_intensities_rel"], dtype=float)

        matched = match_ir_modes(
            pred_freqs_cm=np.asarray(active_freqs, dtype=float),
            pred_intensities=np.asarray(active_intensities, dtype=float),
            ref_freqs_cm=ref_freqs,
            ref_intensities=ref_intensities,
        )

        mode_match_summary = {
            "matched_mode_count": matched["matched_mode_count"],
            "pred_freqs_matched_cm": matched["pred_freqs_matched_cm"],
            "ref_freqs_matched_cm": matched["ref_freqs_matched_cm"],
            "pred_intensities_matched": matched["pred_intensities_matched"],
            "ref_intensities_matched": matched["ref_intensities_matched"],
            "freq_abs_errors_cm": matched["freq_abs_errors_cm"],
            "freq_mae_ir_cm1": matched["freq_mae_ir_cm1"],
            "freq_rmse_ir_cm1": matched["freq_rmse_ir_cm1"],
            "freq_mae_ir_weighted_cm1": matched["freq_mae_ir_weighted_cm1"],
            "intensity_pearson_r": matched["intensity_pearson_r"],
            "intensity_spearman_r": matched["intensity_spearman_r"],
        }

    crystal_mode_comparison = run_optional_crystal_mode_comparison(
        atoms=atoms,
        mace_modes=mace_modes,
        structure=structure,
        output_dir=output_dir,
        enabled=compare_crystal_modes,
        crystal_hess_path=crystal_hess_path,
        freq_out_path=crystal_freq_out_path,
        crystal_structures_root=crystal_structures_root,
        crystal_hessian_units=crystal_hessian_units,
        skip_first=mode_skip_first,
        degeneracy_tol=mode_degeneracy_tol,
        ref_db_path=crystal_db_path,
        use_ref_db=True,
    )

    composite_score = make_composite_score(
        n_imag_modes=int(np.sum(imag_flags)),
        freq_mae_ir_cm1=mode_match_summary["freq_mae_ir_cm1"],
        freq_mae_ir_weighted_cm1=mode_match_summary["freq_mae_ir_weighted_cm1"],
        spectrum_rel_l2=crystal_summary.get("spectrum_rel_l2"),
        intensity_pearson_r=mode_match_summary["intensity_pearson_r"],
    )

    phonopy_summary = {
        "enabled": bool(run_phonopy),
        "implemented": False,
        "plugin": phonopy_plugin,
        "status": "skipped",
    }

    summary = {
        "structure": structure,
        "model_path": model_path,
        "cif_path": cif_path,
        "output_dir": output_dir,
        "device": device,
        "default_dtype": default_dtype,
        "frechet": frechet,
        "fmax": fmax,
        "compare_crystal_modes": compare_crystal_modes,
        "crystal_structures_root": crystal_structures_root,
        "crystal_hessian_units": crystal_hessian_units,
        "mode_skip_first": mode_skip_first,
        "mode_degeneracy_tol": mode_degeneracy_tol,
        "natoms_primitive": natoms,
        "initial_max_force_eV_per_A": initial_max_force,
        "zpe_eV": float(zpe_eV),
        "n_modes_total": int(len(freqs_cm)),
        "n_imag_modes": int(np.sum(imag_flags)),
        "n_physical_modes": int(np.sum((~imag_flags) & (np.asarray(freqs_cm) > 1e-6))),
        "n_ir_active_modes": int(np.sum(active_mask)),
        "optimized_cell": optimized_cell,
        "optimized_scaled_positions": optimized_scaled_positions,
        "bec_summary": bec_summary,
        "crystal_comparison": crystal_summary,
        "phonopy": phonopy_summary,
        "mode_matching": mode_match_summary,
        "crystal_mode_comparison": crystal_mode_comparison,
        "ranking_metrics": {
            "freq_mae_ir_cm1": mode_match_summary["freq_mae_ir_cm1"],
            "freq_rmse_ir_cm1": mode_match_summary["freq_rmse_ir_cm1"],
            "freq_mae_ir_weighted_cm1": mode_match_summary["freq_mae_ir_weighted_cm1"],
            "intensity_pearson_r": mode_match_summary["intensity_pearson_r"],
            "intensity_spearman_r": mode_match_summary["intensity_spearman_r"],
            "spectrum_rel_l2": crystal_summary.get("spectrum_rel_l2"),
            "matched_mode_count": mode_match_summary["matched_mode_count"],
            "composite_score": composite_score,
            "crystal_mode_mean_abs_freq_error_cm1": crystal_mode_comparison.get("mean_abs_freq_error_cm1"),
            "crystal_mode_mean_overlap": crystal_mode_comparison.get("mean_mode_overlap"),
            "crystal_mode_mean_subspace_overlap": crystal_mode_comparison.get("mean_subspace_overlap"),
        },
        "artifacts": {
            "ir_plot": ir_plot_path,
            "mode_overlap_plot": crystal_mode_comparison.get("heatmap_outfile"),
            "group_mode_overlap_plot": crystal_mode_comparison.get("group_heatmap_outfile"),
            "trajectory": traj_path,
            "summary_json": summary_json_path,
            "arrays_npz": npz_path,
        },
        "ref_db": {
            "write_ref_db": bool(write_ref_db),
            "ref_db_path": None if ref_db_path is None else str(ref_db_path),
            "run_id": run_id,
            "dataset_split": dataset_split,
            "sweep_id": sweep_id,
            "write_status": "not_requested",
            "error": None,
        },
    }

    # ------------------------------------------------------------
    # Optional write-back to reference DB
    # ------------------------------------------------------------
    if write_ref_db:
        if ref_db_path is None:
            summary["ref_db"]["write_status"] = "failed"
            summary["ref_db"]["error"] = "write_ref_db=True but ref_db_path=None"
        else:
            if run_id is None:
                run_id_final = model_path.stem
            else:
                run_id_final = str(run_id)

            summary["ref_db"]["run_id"] = run_id_final

            try:
                write_model_evaluation(
                    ref_db_path=ref_db_path,
                    structure=structure,
                    run_id=run_id_final,
                    dataset_split=dataset_split or "ungrouped",
                    sweep_id=sweep_id or "manual",
                    atoms=atoms,
                    bec_raw=bec,
                    bec_asr=bec_asr,
                    frequencies_cm1=np.asarray(freqs_cm, dtype=float),
                    imag_flags=np.asarray(imag_flags, dtype=bool),
                    eigvals_SI=np.asarray(eigvals_SI, dtype=float),
                    eigvecs_mw=np.asarray(eigvecs, dtype=float),
                    intensities=np.asarray(intensities, dtype=float),
                    z_mode=np.asarray(z_mode, dtype=float),
                    nu_grid_cm1=np.asarray(nu_grid, dtype=float),
                    ir_spec=np.asarray(ir_spec, dtype=float),
                    ranking_metrics=summary.get("ranking_metrics", {}),
                    metadata={
                        "model_path": model_path,
                        "output_dir": output_dir,
                        "summary_json": summary_json_path,
                        "arrays_npz": npz_path,
                        "ir_plot": ir_plot_path,
                        "device": device,
                        "default_dtype": default_dtype,
                        "calculator_mode": calculator_mode,
                        "frechet": frechet,
                        "fmax": fmax,
                        "compare_crystal_modes": compare_crystal_modes,
                    },
                )

                summary["ref_db"]["write_status"] = "ok"

            except Exception as exc:
                summary["ref_db"]["write_status"] = "failed"
                summary["ref_db"]["error"] = repr(exc)
                print(f"Reference DB write-back failed: {exc}")

    # save machine-readable arrays
    np.savez(
        npz_path,
        freqs_cm=np.asarray(freqs_cm, dtype=float),
        imag_flags=np.asarray(imag_flags, dtype=bool),
        eigvals_SI=np.asarray(eigvals_SI, dtype=float),
        eigvecs_mw=np.asarray(eigvecs, dtype=float),
        intensities=np.asarray(intensities, dtype=float),
        z_mode=np.asarray(z_mode, dtype=float),
        nu_grid=np.asarray(nu_grid, dtype=float),
        ir_spec=np.asarray(ir_spec, dtype=float),
        active_freqs_cm=np.asarray(active_freqs, dtype=float),
        active_intensities=np.asarray(active_intensities, dtype=float),
        bec_asr=np.asarray(bec_asr, dtype=float),
        crystal_freqs_cm=np.asarray(
            [] if crystal_summary["crystal_freqs_cm"] is None else crystal_summary["crystal_freqs_cm"],
            dtype=float,
        ),
        crystal_intensities_rel=np.asarray(
            [] if crystal_summary["crystal_intensities_rel"] is None else crystal_summary["crystal_intensities_rel"],
            dtype=float,
        ),
        matched_pred_freqs_cm=np.asarray(
            [] if mode_match_summary["pred_freqs_matched_cm"] is None else mode_match_summary["pred_freqs_matched_cm"],
            dtype=float,
        ),
        matched_ref_freqs_cm=np.asarray(
            [] if mode_match_summary["ref_freqs_matched_cm"] is None else mode_match_summary["ref_freqs_matched_cm"],
            dtype=float,
        ),
        matched_pred_intensities=np.asarray(
            [] if mode_match_summary["pred_intensities_matched"] is None else mode_match_summary["pred_intensities_matched"],
            dtype=float,
        ),
        matched_ref_intensities=np.asarray(
            [] if mode_match_summary["ref_intensities_matched"] is None else mode_match_summary["ref_intensities_matched"],
            dtype=float,
        ),
        matched_freq_abs_errors_cm=np.asarray(
            [] if mode_match_summary["freq_abs_errors_cm"] is None else mode_match_summary["freq_abs_errors_cm"],
            dtype=float,
        ),
    )

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2)

    # cleanup
    atoms.calc = None
    del calc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MACE/MACELES model on IR/phonons.")
    parser.add_argument("--model-path", required=True, help="Path to trained model.")
    parser.add_argument("--structure", required=True, help="Structure name, e.g. SiO2.")
    parser.add_argument("--cif-path", default=None, help="Optional explicit CIF path.")
    parser.add_argument("--output-dir", default=None, help="Directory for plots and summaries.")
    parser.add_argument("--crystal-db", default=str(DEFAULT_CRYSTAL_DB), help="Path to CRYSTALreference.h5.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--no-frechet", action="store_true", help="Disable FrechetCellFilter.")
    parser.add_argument("--fmax", type=float, default=1e-11)
    parser.add_argument("--calculator-mode", default="0", choices=["0", "1"])
    parser.add_argument("--compare-crystal-modes", action="store_true", help="Enable explicit CRYSTAL hessfreq eigenvector/subspace mode matching.")
    parser.add_argument("--crystal-hess-path", default=None, help="Optional explicit CRYSTAL *_freq.hessfreq path for mode matching.")
    parser.add_argument("--crystal-freq-out", default=None, help="Optional explicit CRYSTAL *_freq.out path for degeneracy/frequency metadata.")
    parser.add_argument("--crystal-structures-root", default=str(DEFAULT_CRYSTAL_STRUCTURES_ROOT), help="Root containing <structure>/freq/<structure>_freq.hessfreq.")
    parser.add_argument("--crystal-hessian-units", default=DEFAULT_CRYSTAL_HESSIAN_UNITS, help="Units passed to TOOLs.ModeComparison.run_mode_comparison.")
    parser.add_argument("--mode-skip-first", type=int, default=3)
    parser.add_argument("--mode-degeneracy-tol", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    summary = evaluate_model(
        model_path=args.model_path,
        structure=args.structure,
        cif_path=args.cif_path,
        output_dir=args.output_dir,
        crystal_db_path=args.crystal_db,
        device=args.device,
        default_dtype=args.default_dtype,
        frechet=not args.no_frechet,
        fmax=args.fmax,
        calculator_mode=args.calculator_mode,
        compare_crystal_modes=args.compare_crystal_modes,
        crystal_hess_path=args.crystal_hess_path,
        crystal_freq_out_path=args.crystal_freq_out,
        crystal_structures_root=args.crystal_structures_root,
        crystal_hessian_units=args.crystal_hessian_units,
        mode_skip_first=args.mode_skip_first,
        mode_degeneracy_tol=args.mode_degeneracy_tol,
    )

    print("\nEvaluation finished.")
    print(json.dumps(to_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
    