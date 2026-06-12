#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any
from ase import Atoms
import re
import h5py
import spglib
import numpy as np

#################################################################
# Helpers
#################################################################

def parse_hyperparams_from_run_id(run_id: str) -> dict[str, Any]:
    """
    Parse common sweep hyperparameters from run names like:

        MACELES_bs2_ep173_ew1_fw75_rmax4_seed2_SiO2_n1400

    Returns keys only when they are found:
        batch_size, max_epochs, energy_weight, forces_weight,
        r_max, seed, train_size
    """
    run_id = str(run_id)

    patterns = {
        "batch_size": r"(?:^|_)bs(?P<value>[0-9]+)(?:_|$)",
        "max_epochs": r"(?:^|_)ep(?P<value>[0-9]+)(?:_|$)",
        "energy_weight": r"(?:^|_)ew(?P<value>[0-9]+(?:p[0-9]+)?|[0-9]+(?:\.[0-9]+)?)(?:_|$)",
        "forces_weight": r"(?:^|_)fw(?P<value>[0-9]+(?:p[0-9]+)?|[0-9]+(?:\.[0-9]+)?)(?:_|$)",
        "r_max": r"(?:^|_)rmax(?P<value>[0-9]+(?:p[0-9]+)?|[0-9]+(?:\.[0-9]+)?)(?:_|$)",
        "seed": r"(?:^|_)seed(?P<value>[0-9]+)(?:_|$)",
        "train_size": r"(?:^|_)n(?P<value>[0-9]+)(?:_|$)",
    }

    int_keys = {"batch_size", "max_epochs", "seed", "train_size"}
    out: dict[str, Any] = {}

    for key, pattern in patterns.items():
        m = re.search(pattern, run_id)
        if m is None:
            continue

        raw = m.group("value").replace("p", ".")

        if key in int_keys:
            out[key] = int(raw)
        else:
            out[key] = float(raw)

    return out


def format_hyperparams(
    hyperparameters: dict[str, Any] | None,
    *,
    keys: tuple[str, ...] = (
        "r_max",
        "energy_weight",
        "forces_weight",
        "seed",
        "train_size",
        "batch_size",
        "max_epochs",
    ),
) -> str:
    """
    Compact plot-safe hyperparameter string.

    Avoid underscores because your Matplotlib style may use text.usetex=True.
    """
    if not hyperparameters:
        return ""

    labels = {
        "r_max": "rmax",
        "energy_weight": "ew",
        "forces_weight": "fw",
        "seed": "seed",
        "train_size": "n",
        "batch_size": "bs",
        "max_epochs": "ep",
    }

    parts = []

    for key in keys:
        if key not in hyperparameters:
            continue

        value = hyperparameters[key]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and value.is_integer():
            value = int(value)

        parts.append(f"{labels.get(key, key)} {value}")

    return ", ".join(parts)


def fix_label_case(label):
    """Convert an atomic label like FE1 -> Fe1 or SI2 -> Si."""
    match = re.match(r"([A-Za-z]+)([0-9]*)", label)
    if not match:
        return label  # leave unchanged if not matching expected pattern
    elem, idx = match.groups()
    # Only capitalize first letter, lowercase the rest (Fe, Si, Co, etc.)
    elem_fixed = elem.capitalize()
    return elem_fixed


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def overwrite_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def overwrite_dataset(group: h5py.Group, name: str, data, **kwargs):
    if name in group:
        del group[name]
    return group.create_dataset(name, data=data, **kwargs)


def as_hdf5_string_array(values):
    return np.asarray(values).astype("S")


def clean_attr_value(value):
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def standardize_to_primitive(atoms, no_idealize=False):
    cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    prim = spglib.standardize_cell(cell, to_primitive=True, no_idealize=no_idealize)
    if prim is None:
        raise ValueError("spglib.standardize_cell returned None")
    return Atoms(
        numbers=prim[2],
        scaled_positions=prim[1],
        cell=prim[0],
        pbc=True,
    )


def model_evaluation_group_path(
    structure: str,
    dataset_split: str,
    sweep_id: str,
    run_id: str,
) -> str:
    return (
        f"structures/{structure}/evaluations/"
        f"{dataset_split}/{sweep_id}/{run_id}"
    )


def model_evaluation_exists(
    ref_db_path: str | Path,
    structure: str,
    run_id: str,
    dataset_split: str = "ungrouped",
    sweep_id: str = "manual",
    required_datasets: list[str] | None = None,
) -> bool:
    if required_datasets is None:
        required_datasets = [
            "frequencies_cm1",
            "imag_flags",
            "intensities",
            "z_mode",
            "ir_spectrum/nu_grid_cm1",
            "ir_spectrum/intensity_relative",
        ]

    path = model_evaluation_group_path(
        structure=structure,
        dataset_split=dataset_split,
        sweep_id=sweep_id,
        run_id=run_id,
    )

    ref_db_path = Path(ref_db_path)

    if not ref_db_path.exists():
        return False

    with h5py.File(ref_db_path, "r") as h5:
        if path not in h5:
            return False

        grp = h5[path]

        for name in required_datasets:
            if name not in grp:
                return False

        return True


def update_model_ranking_metrics(
    ref_db_path,
    structure,
    run_id,
    dataset_split,
    sweep_id,
    ranking_metrics,
):
    path = model_evaluation_group_path(structure, dataset_split, sweep_id, run_id)

    with h5py.File(ref_db_path, "a") as h5:
        if path not in h5:
            raise KeyError(f"Missing model evaluation group: {path}")

        eg = h5[path]

        if "ranking_metrics" in eg:
            del eg["ranking_metrics"]

        rg = eg.create_group("ranking_metrics")
        write_attrs(rg, ranking_metrics)


def backfill_hyperparameters_from_run_ids(
    ref_db_path: str | Path,
    *,
    structure: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Backfill /hyperparameters attrs for old model evaluations by parsing run_id.

    This assumes parse_hyperparams_from_run_id(run_id) exists.
    """
    ref_db_path = Path(ref_db_path)

    with h5py.File(ref_db_path, "a") as h5:
        structures_root = h5["structures"]

        structure_names = [structure] if structure is not None else list(structures_root.keys())

        for structure_name in structure_names:
            eval_root_path = f"structures/{structure_name}/evaluations"

            if eval_root_path not in h5:
                continue

            eval_root = h5[eval_root_path]

            for split_name, split_group in eval_root.items():
                for sweep_name, sweep_group in split_group.items():
                    for run_id, run_group in sweep_group.items():
                        if "hyperparameters" in run_group and not overwrite:
                            continue

                        hp = parse_hyperparams_from_run_id(run_id)

                        if not hp:
                            print(f"SKIP no parsed hyperparams: {structure_name}/{run_id}")
                            continue

                        write_hyperparameters(run_group, hp)

                        print(
                            f"backfilled {structure_name} / {split_name} / "
                            f"{sweep_name} / {run_id}: {hp}"
                        )


#################################################################
# Writers
#################################################################

def write_attrs(group: h5py.Group, attrs: dict[str, Any] | None):
    if not attrs:
        return
    for key, value in attrs.items():
        value = clean_attr_value(value)
        if value is not None:
            group.attrs[str(key)] = value


def write_geometry(group: h5py.Group, atoms):
    overwrite_dataset(group, "atomic_numbers", np.asarray(atoms.numbers, dtype=np.int32))

    overwrite_dataset(
        group,
        "symbols",
        as_hdf5_string_array(atoms.get_chemical_symbols()),
    )

    overwrite_dataset(group, "positions_A", np.asarray(atoms.positions, dtype=np.float64))
    overwrite_dataset(group, "cell_A", np.asarray(atoms.cell.array, dtype=np.float64))

    overwrite_dataset(
        group,
        "scaled_positions",
        np.asarray(atoms.get_scaled_positions(wrap=True), dtype=np.float64),
    )

    group.attrs["geometry_source_convention"] = "CRYSTAL freq.out primitive cell order"
    group.attrs["position_units"] = "Angstrom"
    group.attrs["cell_units"] = "Angstrom"
    group.attrs["n_atoms"] = int(len(atoms))


def write_crystal_reference(
    ref_db_path: str | Path,
    structure: str,
    *,
    atoms=None,
    born_charges=None,
    born_species=None,
    dielectric_tensor=None,
    hessian_cart_eV_A2=None,
    hessian_mw_SI=None,
    eigvals_SI=None,
    eigvecs_mw=None,
    frequencies_cm1=None,
    ir_frequencies_cm1=None,
    imag_flags=None,
    intensities_km_mol=None,
    degeneracies=None,
    irreps=None,
    metadata: dict[str, Any] | None = None,
):
    ref_db_path = ensure_parent(ref_db_path)

    with h5py.File(ref_db_path, "a") as h5:
        root = h5.require_group("structures")
        sg = root.require_group(structure)
        cg = overwrite_group(sg, "crystal")

        if atoms is not None:
            geom = cg.create_group("geometry")
            write_geometry(geom, atoms)

            cg.attrs["atom_order_source"] = "CRYSTAL freq.out primitive cell"
            cg.attrs["eigvecs_atom_order"] = "same as /geometry"
            cg.attrs["hessian_atom_order"] = "same as /geometry"
            cg.attrs["eigvecs_convention"] = "mass_weighted"

        arrays = {
            "born_charges": born_charges,
            "born_species": born_species,
            "dielectric_tensor": dielectric_tensor,
            "hessian_cart_eV_A2": hessian_cart_eV_A2,
            "hessian_mw_SI": hessian_mw_SI,
            "eigvals_SI": eigvals_SI,
            "eigvecs_mw": eigvecs_mw,
            "frequencies_cm1": frequencies_cm1,
            "ir_frequencies_cm1": ir_frequencies_cm1,
            "imag_flags": imag_flags,
            "intensities_km_mol": intensities_km_mol,
            "degeneracies": degeneracies,
        }

        for name, value in arrays.items():
            if value is not None:
                overwrite_dataset(cg, name, np.asarray(value))

        if irreps is not None:
            overwrite_dataset(cg, "irreps", as_hdf5_string_array(irreps))

        write_attrs(cg, metadata)

        cg.attrs["schema"] = "crystal_reference_v1"


def write_model_evaluation(
    ref_db_path: str | Path,
    structure: str,
    run_id: str,
    *,
    dataset_split="ungrouped",
    sweep_id="manual",
    atoms=None,
    bec_raw=None,
    bec_asr=None,
    frequencies_cm1=None,
    imag_flags=None,
    eigvals_SI=None,
    eigvecs_mw=None,
    intensities=None,
    z_mode=None,
    nu_grid_cm1=None,
    ir_spec=None,
    mode_matching=None,
    ir_matching=None,
    ranking_metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    hyperparameters: dict[str, Any] | None = None,
):
    ref_db_path = ensure_parent(ref_db_path)

    with h5py.File(ref_db_path, "a") as h5:
        root = h5.require_group("structures")
        sg = root.require_group(structure)

        eg_root = sg.require_group("evaluations")
        split_group = eg_root.require_group(str(dataset_split))
        sweep_group = split_group.require_group(str(sweep_id))
        eg = overwrite_group(sweep_group, str(run_id))

        split_group.attrs["group_type"] = "dataset_split"
        sweep_group.attrs["group_type"] = "sweep"
        eg.attrs["group_type"] = "model_run"

        eg.attrs["dataset_split"] = str(dataset_split)
        eg.attrs["sweep_id"] = str(sweep_id)

        if atoms is not None:
            geom = eg.create_group("geometry_optimized")
            write_geometry(geom, atoms)

        arrays = {
            "bec_raw": bec_raw,
            "bec_asr": bec_asr,
            "frequencies_cm1": frequencies_cm1,
            "imag_flags": imag_flags,
            "eigvals_SI": eigvals_SI,
            "eigvecs_mw": eigvecs_mw,
            "intensities": intensities,
            "z_mode": z_mode,
        }

        for name, value in arrays.items():
            if value is not None:
                overwrite_dataset(eg, name, np.asarray(value))

        if nu_grid_cm1 is not None or ir_spec is not None:
            irg = eg.create_group("ir_spectrum")
            if nu_grid_cm1 is not None:
                overwrite_dataset(irg, "nu_grid_cm1", np.asarray(nu_grid_cm1))
            if ir_spec is not None:
                overwrite_dataset(irg, "intensity_relative", np.asarray(ir_spec))

        write_mode_matching(eg, mode_matching)
        write_ir_matching(eg, ir_matching)

        if ranking_metrics is not None:
            rg = eg.create_group("ranking_metrics")
            write_attrs(rg, ranking_metrics)

        write_hyperparameters(eg, hyperparameters)
        write_attrs(eg, metadata)

        eg.attrs["schema"] = "model_evaluation_v1"


def write_scalar_summary_group(group: h5py.Group, name: str, summary: dict[str, Any] | None):
    if not summary:
        return

    sg = overwrite_group(group, name)
    write_attrs(sg, summary)


def write_mode_matching(group: h5py.Group, mode_matching: dict[str, Any] | None):
    if not mode_matching:
        return

    mg = overwrite_group(group, "mode_matching")

    # Metadata attrs
    for key in [
        "source",
        "skip_first",
        "degeneracy_tol",
    ]:
        if key in mode_matching:
            value = clean_attr_value(mode_matching[key])
            if value is not None:
                mg.attrs[key] = value

    crystal = mode_matching.get("crystal", {})
    for key in [
        "origin_shift",
        "max_atom_mismatch",
    ]:
        if key in crystal:
            value = crystal[key]
            if value is not None:
                overwrite_dataset(mg, key, np.asarray(value))

    if "atom_permutation" in crystal and crystal["atom_permutation"] is not None:
        overwrite_dataset(
            mg,
            "atom_permutation",
            np.asarray(crystal["atom_permutation"], dtype=np.int32),
        )

    # Overlap matrices
    matrices = overwrite_group(mg, "overlap_matrices")

    for key in [
        "overlap_full",
        "overlap_cut",
        "group_overlap_matrix",
    ]:
        if key in mode_matching and mode_matching[key] is not None:
            overwrite_dataset(
                matrices,
                key,
                np.asarray(mode_matching[key], dtype=np.float64),
            )

    # Mode matches table
    matches = mode_matching.get("matches", []) or []
    if matches:
        mm = overwrite_group(mg, "mode_matches")

        overwrite_dataset(mm, "ref_mode_index", np.asarray([m["mode_ref"] for m in matches], dtype=np.int32))
        overwrite_dataset(mm, "model_mode_index", np.asarray([m["mode_test"] for m in matches], dtype=np.int32))
        overwrite_dataset(mm, "ref_freq_cm1", np.asarray([m["freq_ref"] for m in matches], dtype=np.float64))
        overwrite_dataset(mm, "model_freq_cm1", np.asarray([m["freq_test"] for m in matches], dtype=np.float64))
        overwrite_dataset(mm, "delta_cm1", np.asarray([m["delta_cm1"] for m in matches], dtype=np.float64))
        overwrite_dataset(mm, "abs_delta_cm1", np.asarray([abs(m["delta_cm1"]) for m in matches], dtype=np.float64))
        overwrite_dataset(mm, "overlap", np.asarray([m["overlap"] for m in matches], dtype=np.float64))

    # Degenerate/subspace group matches table
    groups = mode_matching.get("subgroups", []) or []
    if groups:
        gg = overwrite_group(mg, "group_matches")

        ref_starts = []
        ref_ends = []
        model_starts = []
        model_ends = []

        for g in groups:
            ref_modes = np.asarray(g["ref_modes"], dtype=int)
            model_modes = np.asarray(g["test_modes"], dtype=int)

            ref_starts.append(int(ref_modes[0]))
            ref_ends.append(int(ref_modes[-1]))
            model_starts.append(int(model_modes[0]))
            model_ends.append(int(model_modes[-1]))

        overwrite_dataset(gg, "ref_group_index", np.asarray([g["group_ref_index"] for g in groups], dtype=np.int32))
        overwrite_dataset(gg, "model_group_index", np.asarray([g["group_test_index"] for g in groups], dtype=np.int32))
        overwrite_dataset(gg, "ref_group_start", np.asarray(ref_starts, dtype=np.int32))
        overwrite_dataset(gg, "ref_group_end", np.asarray(ref_ends, dtype=np.int32))
        overwrite_dataset(gg, "model_group_start", np.asarray(model_starts, dtype=np.int32))
        overwrite_dataset(gg, "model_group_end", np.asarray(model_ends, dtype=np.int32))
        overwrite_dataset(gg, "ref_freq_mean_cm1", np.asarray([g["ref_freq_mean"] for g in groups], dtype=np.float64))
        overwrite_dataset(gg, "model_freq_mean_cm1", np.asarray([g["test_freq_mean"] for g in groups], dtype=np.float64))
        overwrite_dataset(gg, "delta_cm1", np.asarray([g["delta_cm1"] for g in groups], dtype=np.float64))
        overwrite_dataset(gg, "subspace_overlap", np.asarray([g["subspace_overlap"] for g in groups], dtype=np.float64))

    # Scalar summaries
    mode_overlaps = [m["overlap"] for m in matches] if matches else []
    subspace_overlaps = [g["subspace_overlap"] for g in groups] if groups else []

    summary = {}

    if mode_overlaps:
        summary["mean_mode_overlap"] = float(np.mean(mode_overlaps))
        summary["min_mode_overlap"] = float(np.min(mode_overlaps))

    if subspace_overlaps:
        summary["mean_subspace_overlap"] = float(np.mean(subspace_overlaps))
        summary["min_subspace_overlap"] = float(np.min(subspace_overlaps))

    if "overlap_cut" in mode_matching and mode_matching["overlap_cut"] is not None:
        O = np.asarray(mode_matching["overlap_cut"], dtype=float)
        if O.ndim == 2 and O.size > 0:
            n_diag = min(O.shape)
            diag = np.diag(O[:n_diag, :n_diag])
            summary["diagonal_overlap_mean"] = float(np.mean(diag))

            offdiag = O.copy()
            for i in range(n_diag):
                offdiag[i, i] = np.nan

            summary["offdiag_leakage_mean"] = float(np.nanmean(offdiag))

    write_scalar_summary_group(mg, "summaries", summary)


def write_ir_matching(group: h5py.Group, ir_matching: dict[str, Any] | None):
    if not ir_matching:
        return

    ig = overwrite_group(group, "ir_matching")

    pred_freqs = ir_matching.get("pred_freqs_matched_cm")
    ref_freqs = ir_matching.get("ref_freqs_matched_cm")
    pred_intensities = ir_matching.get("pred_intensities_matched")
    ref_intensities = ir_matching.get("ref_intensities_matched")
    abs_errors = ir_matching.get("freq_abs_errors_cm")

    if pred_freqs is not None and ref_freqs is not None:
        pg = overwrite_group(ig, "matched_peaks")

        pred_freqs = np.asarray(pred_freqs, dtype=np.float64)
        ref_freqs = np.asarray(ref_freqs, dtype=np.float64)

        overwrite_dataset(pg, "model_freq_cm1", pred_freqs)
        overwrite_dataset(pg, "ref_freq_cm1", ref_freqs)
        overwrite_dataset(pg, "delta_cm1", pred_freqs - ref_freqs)

        if abs_errors is not None:
            overwrite_dataset(pg, "abs_delta_cm1", np.asarray(abs_errors, dtype=np.float64))
        else:
            overwrite_dataset(pg, "abs_delta_cm1", np.abs(pred_freqs - ref_freqs))

        if pred_intensities is not None:
            overwrite_dataset(pg, "model_intensity", np.asarray(pred_intensities, dtype=np.float64))

        if ref_intensities is not None:
            overwrite_dataset(pg, "ref_intensity", np.asarray(ref_intensities, dtype=np.float64))

    summary = {
        "matched_mode_count": ir_matching.get("matched_mode_count"),
        "freq_mae_ir_cm1": ir_matching.get("freq_mae_ir_cm1"),
        "freq_rmse_ir_cm1": ir_matching.get("freq_rmse_ir_cm1"),
        "freq_mae_ir_weighted_cm1": ir_matching.get("freq_mae_ir_weighted_cm1"),
        "intensity_pearson_r": ir_matching.get("intensity_pearson_r"),
        "intensity_spearman_r": ir_matching.get("intensity_spearman_r"),
    }

    write_scalar_summary_group(ig, "summaries", summary)


def write_hyperparameters(group: h5py.Group, hyperparameters: dict[str, Any] | None):
    """
    Store training/sweep hyperparameters as attrs in a dedicated subgroup.

    Expected examples:
        r_max
        energy_weight
        forces_weight
        seed
        batch_size
        max_epochs
        train_size
    """
    if not hyperparameters:
        return

    hg = overwrite_group(group, "hyperparameters")
    write_attrs(hg, hyperparameters)


#################################################################
# Readers
#################################################################

def read_crystal_reference(ref_db_path: str | Path, structure: str) -> dict:
    ref_db_path = Path(ref_db_path)

    with h5py.File(ref_db_path, "r") as h5:
        path = f"structures/{structure}/crystal"

        if path not in h5:
            raise KeyError(f"Missing CRYSTAL reference: {path}")

        cg = h5[path]

        out = {
            "attrs": dict(cg.attrs),
        }

        for key, obj in cg.items():
            if isinstance(obj, h5py.Dataset):
                out[key] = obj[()]
            elif isinstance(obj, h5py.Group):
                out[key] = {
                    sub_key: sub_obj[()]
                    for sub_key, sub_obj in obj.items()
                    if isinstance(sub_obj, h5py.Dataset)
                }

        return out


def read_crystal_ir_reference(ref_db_path: str | Path, structure: str) -> tuple[np.ndarray, np.ndarray]:
    ref = read_crystal_reference(ref_db_path, structure)

    if "ir_frequencies_cm1" not in ref:
        raise KeyError(f"No ir_frequencies_cm1 stored for {structure}")

    if "intensities_km_mol" not in ref:
        raise KeyError(f"No intensities_km_mol stored for {structure}")

    freqs = np.asarray(ref["ir_frequencies_cm1"], dtype=float)
    intensities = np.asarray(ref["intensities_km_mol"], dtype=float)

    if len(freqs) != len(intensities):
        raise ValueError(
            f"IR frequency/intensity length mismatch for {structure}: "
            f"{len(freqs)} vs {len(intensities)}"
        )

    return freqs, intensities


def read_crystal_modes(ref_db_path: str | Path, structure: str) -> dict:
    """
    Read CRYSTAL Gamma-point mode data and matching reference geometry
    from the reference DB.

    Expected path:
        /structures/<structure>/crystal
    """
    ref_db_path = Path(ref_db_path)

    with h5py.File(ref_db_path, "r") as h5:
        path = f"structures/{structure}/crystal"

        if path not in h5:
            raise KeyError(f"Missing CRYSTAL reference group: {path}")

        cg = h5[path]

        required = ["frequencies_cm1", "eigvecs_mw"]
        missing = [key for key in required if key not in cg]
        if missing:
            raise KeyError(
                f"Missing required CRYSTAL mode datasets for {structure}: {missing}"
            )

        out = {
            "freqs_cm": np.asarray(cg["frequencies_cm1"][()], dtype=float),
            "eigvecs_mw": np.asarray(cg["eigvecs_mw"][()], dtype=float),
            "attrs": dict(cg.attrs),
        }

        optional = [
            "eigvals_SI",
            "imag_flags",
            "hessian_mw_SI",
            "hessian_cart_eV_A2",
        ]

        for key in optional:
            if key in cg:
                out[key] = cg[key][()]

        if "geometry" in cg:
            geom = cg["geometry"]

            out["geometry"] = {
                "attrs": dict(geom.attrs),
            }

            for key in [
                "atomic_numbers",
                "symbols",
                "positions_A",
                "cell_A",
                "scaled_positions",
            ]:
                if key not in geom:
                    continue

                value = geom[key][()]

                if key == "symbols":
                    value = [
                        s.decode("utf-8") if isinstance(s, bytes) else str(s)
                        for s in value
                    ]

                out["geometry"][key] = value

            # Convenience flat keys for mode_matching.py
            if "atomic_numbers" in out["geometry"]:
                out["atomic_numbers"] = out["geometry"]["atomic_numbers"]
            if "symbols" in out["geometry"]:
                out["symbols"] = out["geometry"]["symbols"]
            if "positions_A" in out["geometry"]:
                out["positions_A"] = out["geometry"]["positions_A"]
            if "cell_A" in out["geometry"]:
                out["cell_A"] = out["geometry"]["cell_A"]
            if "scaled_positions" in out["geometry"]:
                out["scaled_positions"] = out["geometry"]["scaled_positions"]

        return out


def read_crystal_lattice_from_freq_out(freq_out_file):
    """
    Parse the 'DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)' block
    from a CRYSTAL output file.

    Returns
    -------
    cell : (3, 3) ndarray
    """
    with open(freq_out_file, "r") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS" in line:
            start = i
            break

    if start is None:
        raise ValueError("Direct lattice vector block not found in freq.out")

    vecs = []
    float_re = r"([+-]?\d+\.\d+E[+-]?\d+)"
    pat = re.compile(rf"^\s*{float_re}\s+{float_re}\s+{float_re}\s*$")

    for line in lines[start + 1:]:
        m = pat.match(line)
        if m:
            vecs.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            if len(vecs) == 3:
                break

    if len(vecs) != 3:
        raise ValueError("Could not parse 3 direct lattice vectors from freq.out")

    return np.array(vecs, dtype=float)


def read_crystal_primitive_atoms_from_freq_out(freq_out_file):
    """
    Parse the 'CARTESIAN COORDINATES - PRIMITIVE CELL' block from a CRYSTAL freq.out
    and return an ASE Atoms object in that exact printed order.

    Parameters
    ----------
    freq_out_file : str
    cell : (3, 3) array-like
        Cell to assign to the returned Atoms object.
        Use the same primitive cell as in your MACE workflow.

    Returns
    -------
    atoms : ase.Atoms
    """
    with open(freq_out_file, "r") as f:
        lines = f.readlines()
    
    # lattice
    cell = read_crystal_lattice_from_freq_out(freq_out_file)

    start = None
    for i, line in enumerate(lines):
        if "CARTESIAN COORDINATES - PRIMITIVE CELL" in line:
            start = i
            break
    if start is None:
        raise ValueError("Primitive-cell coordinate block not found in freq.out")
    atom_lines = []
    pattern = re.compile(
        r"^\s*\d+\s+\d+\s+([A-Za-z]+)\s+"
        r"([+-]?\d+\.\d+E[+-]?\d+)\s+"
        r"([+-]?\d+\.\d+E[+-]?\d+)\s+"
        r"([+-]?\d+\.\d+E[+-]?\d+)"
    )
    for line in lines[start + 1:]:
        m = pattern.match(line)
        if m is not None:
            sym = fix_label_case(m.group(1))
            x = float(m.group(2))
            y = float(m.group(3))
            z = float(m.group(4))
            atom_lines.append((sym, [x, y, z]))
        elif atom_lines:
            break
    
    if not atom_lines:
        raise ValueError("No atoms parsed from primitive-cell block")

    symbols = [a[0] for a in atom_lines]
    positions = np.array([a[1] for a in atom_lines], dtype=float)
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    # atoms = standardize_to_primitive(atoms, no_idealize=False)
    return atoms


def read_crystal_hessfreq_flat(hessfreq_file: str | Path, n_atoms: int) -> np.ndarray:
    """
    Read CRYSTAL .hessfreq written as a flat stream of whitespace-separated
    numbers with arbitrary line breaks.

    Returns
    -------
    H_cart : (3N, 3N) ndarray
        Cartesian Hessian in CRYSTAL units (typically Hartree/Bohr^2).
    """
    hessfreq_file = Path(hessfreq_file)
    dim = 3 * n_atoms
    expected = dim * dim

    with open(hessfreq_file, "r") as f:
        tokens = f.read().split()

    flat = np.array(
        [float(x.replace("D", "E").replace("d", "e")) for x in tokens],
        dtype=float,
    )

    if flat.size != expected:
        raise ValueError(
            f"Expected {expected} Hessian entries for {n_atoms} atoms "
            f"({dim}x{dim}), but found {flat.size} in {hessfreq_file}."
        )

    H_cart = flat.reshape(dim, dim)
    H_cart = 0.5 * (H_cart + H_cart.T)
    return H_cart


def _read_attrs_if_exists(group: h5py.Group, name: str) -> dict[str, Any]:
    if name not in group:
        return {}
    return dict(group[name].attrs)


def read_model_evaluation(
    ref_db_path: str | Path,
    structure: str,
    run_id: str,
    dataset_split: str = "ungrouped",
    sweep_id: str = "manual",
) -> dict[str, Any]:
    path = model_evaluation_group_path(
        structure=structure,
        dataset_split=dataset_split,
        sweep_id=sweep_id,
        run_id=run_id,
    )

    with h5py.File(ref_db_path, "r") as h5:
        if path not in h5:
            raise KeyError(f"Missing model evaluation group: {path}")

        eg = h5[path]

        out: dict[str, Any] = {
            "attrs": dict(eg.attrs),
            "frequencies_cm1": np.asarray(eg["frequencies_cm1"][()], dtype=float),
            "imag_flags": np.asarray(eg["imag_flags"][()], dtype=bool),
            "intensities": np.asarray(eg["intensities"][()], dtype=float),
            "z_mode": np.asarray(eg["z_mode"][()], dtype=float),
            "nu_grid_cm1": np.asarray(
                eg["ir_spectrum/nu_grid_cm1"][()],
                dtype=float,
            ),
            "ir_spec": np.asarray(
                eg["ir_spectrum/intensity_relative"][()],
                dtype=float,
            ),
            "ranking_metrics": _read_attrs_if_exists(eg, "ranking_metrics"),
            "hyperparameters": read_hyperparameters_from_group(eg, run_id),
        }

        optional_arrays = [
            "bec_raw",
            "bec_asr",
            "eigvals_SI",
            "eigvecs_mw",
        ]

        for key in optional_arrays:
            if key in eg:
                out[key] = np.asarray(eg[key][()])

        if "geometry_optimized" in eg:
            geom = eg["geometry_optimized"]
            out["geometry_optimized"] = {
                "atomic_numbers": np.asarray(geom["atomic_numbers"][()], dtype=int),
                "positions_A": np.asarray(geom["positions_A"][()], dtype=float),
                "cell_A": np.asarray(geom["cell_A"][()], dtype=float),
                "scaled_positions": np.asarray(
                    geom["scaled_positions"][()],
                    dtype=float,
                ),
            }

        return out
    

def read_mode_overlap(
    ref_db_path: str | Path,
    structure: str,
    run_id: str,
    dataset_split: str = "ungrouped",
    sweep_id: str = "manual",
) -> dict[str, Any]:
    """
    Read cached mode-overlap data from the model evaluation DB entry.

    Returns
    -------
    dict
        Contains:
            overlap_cut
            group_overlap_matrix
            group_matches
    """

    path = model_evaluation_group_path(
        structure=structure,
        dataset_split=dataset_split,
        sweep_id=sweep_id,
        run_id=run_id,
    )

    with h5py.File(ref_db_path, "r") as h5:
        if path not in h5:
            raise KeyError(f"Missing model evaluation group: {path}")

        eg = h5[path]

        if "mode_matching" not in eg:
            raise KeyError(f"No mode_matching group in: {path}")

        mg = eg["mode_matching"]

        out = {}

        # ------------------------------------------------------------
        # overlap matrices
        # ------------------------------------------------------------
        if "overlap_matrices" in mg:
            omg = mg["overlap_matrices"]

            if "overlap_cut" in omg:
                out["overlap_cut"] = np.asarray(
                    omg["overlap_cut"][()],
                    dtype=float,
                )

            if "group_overlap_matrix" in omg:
                out["group_overlap_matrix"] = np.asarray(
                    omg["group_overlap_matrix"][()],
                    dtype=float,
                )

        # ------------------------------------------------------------
        # group matches
        # ------------------------------------------------------------
        out["group_matches"] = []

        if "group_matches" in mg:
            gg = mg["group_matches"]

            ref_start = np.asarray(gg["ref_group_start"][()], dtype=int)
            ref_end = np.asarray(gg["ref_group_end"][()], dtype=int)

            model_start = np.asarray(gg["model_group_start"][()], dtype=int)
            model_end = np.asarray(gg["model_group_end"][()], dtype=int)

            for i in range(len(ref_start)):
                out["group_matches"].append({
                    "ref_modes": np.arange(
                        ref_start[i],
                        ref_end[i] + 1,
                    ),
                    "test_modes": np.arange(
                        model_start[i],
                        model_end[i] + 1,
                    ),
                })

        return out
    

def read_hyperparameters_from_group(group: h5py.Group, run_id: str | None = None) -> dict[str, Any]:
    """
    Read stored hyperparameters from a model evaluation group.

    Priority:
        1. /hyperparameters attrs
        2. selected run-level attrs, for backwards compatibility
        3. optional run_id parsing fallback, if parse_hyperparams_from_run_id exists
    """
    hp: dict[str, Any] = {}

    if "hyperparameters" in group:
        hp.update(dict(group["hyperparameters"].attrs))

    # Backwards compatibility if some params were stored as attrs.
    attr_aliases = {
        "r_max": ["r_max", "rmax"],
        "energy_weight": ["energy_weight", "ew"],
        "forces_weight": ["forces_weight", "fw"],
        "seed": ["seed"],
        "batch_size": ["batch_size", "bs"],
        "max_epochs": ["max_epochs", "epochs", "ep"],
        "train_size": ["train_size", "n_train", "dataset_size"],
    }

    for canonical, aliases in attr_aliases.items():
        if canonical in hp:
            continue
        for key in aliases:
            if key in group.attrs:
                hp[canonical] = group.attrs[key]
                break

    # Optional fallback for old DBs. Only use if you added this parser.
    if run_id is not None and "parse_hyperparams_from_run_id" in globals():
        parsed = parse_hyperparams_from_run_id(run_id)
        for key, value in parsed.items():
            hp.setdefault(key, value)

    return hp


def _metric_sort_direction(metric: str) -> bool:
    """
    Return True if lower metric values are better.
    """
    lower_is_better = {
        "composite_score",
        "spectrum_rel_l2",
        "freq_mae_ir_cm1",
        "freq_rmse_ir_cm1",
        "freq_mae_ir_weighted_cm1",
        "peak_position_score",
        "peak_count_score",
    }

    higher_is_better = {
        "intensity_pearson_r",
        "intensity_spearman_r",
        "mean_mode_overlap",
        "min_mode_overlap",
        "mean_subspace_overlap",
        "min_subspace_overlap",
        "diagonal_overlap_mean",
    }

    if metric in lower_is_better:
        return True

    if metric in higher_is_better:
        return False

    return True


def list_model_evaluations_for_structure(
    ref_db_path: str | Path,
    structure: str,
    *,
    metric: str = "composite_score",
    require_ir_spectrum: bool = True,
    require_metric: bool = True,
    sort: bool = True,
) -> list[dict[str, Any]]:
    """
    List cached model evaluations for one structure.

    Returns lightweight rows with split/sweep/run identifiers, ranking metrics,
    and hyperparameters. The full spectra are then loaded with
    read_model_evaluation(...).
    """
    ref_db_path = Path(ref_db_path)
    rows: list[dict[str, Any]] = []

    with h5py.File(ref_db_path, "r") as h5:
        eval_root_path = f"structures/{structure}/evaluations"

        if eval_root_path not in h5:
            raise KeyError(f"Missing evaluations group: {eval_root_path}")

        eval_root = h5[eval_root_path]

        for dataset_split, split_group in eval_root.items():
            if not isinstance(split_group, h5py.Group):
                continue

            for sweep_id, sweep_group in split_group.items():
                if not isinstance(sweep_group, h5py.Group):
                    continue

                for run_id, run_group in sweep_group.items():
                    if not isinstance(run_group, h5py.Group):
                        continue

                    if require_ir_spectrum:
                        has_ir = (
                            "ir_spectrum" in run_group
                            and "nu_grid_cm1" in run_group["ir_spectrum"]
                            and "intensity_relative" in run_group["ir_spectrum"]
                        )
                        if not has_ir:
                            continue

                    ranking_metrics = {}
                    if "ranking_metrics" in run_group:
                        ranking_metrics = dict(run_group["ranking_metrics"].attrs)

                    metric_value = ranking_metrics.get(metric, None)

                    if metric_value is None and metric in run_group.attrs:
                        metric_value = run_group.attrs[metric]

                    if metric_value is None:
                        if require_metric:
                            continue
                    else:
                        metric_value = float(metric_value)

                    # Preferred: stored hyperparameters.
                    # Fallback logic should live in read_hyperparameters_from_group(...).
                    hyperparameters = read_hyperparameters_from_group(
                        run_group,
                        run_id=str(run_id),
                    )

                    rows.append({
                        "structure": structure,
                        "dataset_split": str(dataset_split),
                        "sweep_id": str(sweep_id),
                        "run_id": str(run_id),
                        "metric": metric,
                        "metric_value": metric_value,
                        "ranking_metrics": ranking_metrics,
                        "hyperparameters": hyperparameters,
                        "hyperparam_label": format_hyperparams(hyperparameters),
                        "attrs": dict(run_group.attrs),
                    })

    if sort:
        lower_is_better = _metric_sort_direction(metric)

        rows = sorted(
            rows,
            key=lambda r: (
                np.inf if r["metric_value"] is None else r["metric_value"]
            ),
            reverse=not lower_is_better,
        )

    return rows