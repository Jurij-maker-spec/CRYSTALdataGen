#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any
from ase import Atoms
import re
import h5py
import spglib
import numpy as np

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
    ranking_metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
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

        if ranking_metrics is not None:
            rg = eg.create_group("ranking_metrics")
            write_attrs(rg, ranking_metrics)

        write_attrs(eg, metadata)

        eg.attrs["schema"] = "model_evaluation_v1"


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

