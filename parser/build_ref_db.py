#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
from ase.io import read

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent

while not (ROOT / "util").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not find project root containing 'util'")
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))

from util.crystal_parser import CrystalOutputParser
from util.mode_matching import read_crystal_hessfreq_flat, read_crystal_primitive_atoms_from_freq_out
from util.phonons import diagonalize_hessian
from util.ref_db import write_crystal_reference


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
HARTREE_PER_BOHR2_TO_EV_PER_A2 = HARTREE_TO_EV / (BOHR_TO_ANG**2)


def find_single_outfile(folder: Path) -> Path | None:
    if not folder.exists():
        return None

    outfiles = sorted(
        p for p in folder.glob("*.out")
        if not p.name.lower().startswith("slurm")
    )

    if not outfiles:
        return None

    preferred = [p for p in outfiles if "_freq" in p.name.lower()]
    if preferred:
        return preferred[0]

    return outfiles[0]


def find_hessfreq_file(folder: Path) -> Path | None:
    if not folder.exists():
        return None

    files = sorted(folder.glob("*.hessfreq"))
    return files[0] if files else None


def find_clean_cif(structure_dir: Path) -> Path | None:
    candidates = sorted(structure_dir.glob("*geoopt_clean.cif"))
    if candidates:
        return candidates[0]

    candidates = sorted(structure_dir.glob("*.cif"))
    if candidates:
        return candidates[0]

    return None


def load_reference_atoms(structure_dir: Path, freq_out: Path | None = None):
    """
    Load the atom ordering/cell compatible with CRYSTAL freq/hessfreq.

    Prefer the primitive cell printed in freq.out, because .hessfreq is written
    for that cell. Fall back to CIF only if freq.out parsing fails.
    """
    if freq_out is not None:
        try:
            atoms = read_crystal_primitive_atoms_from_freq_out(freq_out)
            atoms.set_pbc(True)
            return atoms, freq_out
        except Exception as exc:
            print(f"  warning: primitive atoms from freq.out failed: {exc}")

    clean_cif = find_clean_cif(structure_dir)
    if clean_cif is not None:
        atoms = read(clean_cif)
        atoms.set_pbc(True)
        return atoms, clean_cif

    raise FileNotFoundError(f"No usable reference structure found in {structure_dir}")


def parse_optional_born(parser: CrystalOutputParser):
    try:
        born, species = parser.get_born_charges()
        if isinstance(born, int):
            return None, None
        return np.asarray(born, dtype=float), np.asarray(species, dtype=int)
    except Exception as exc:
        print(f"  warning: BEC parse failed: {exc}")
        return None, None


def parse_optional_dielectric(parser: CrystalOutputParser):
    try:
        return np.asarray(parser.get_dielectric_tensor(), dtype=float)
    except Exception as exc:
        print(f"  warning: dielectric parse failed: {exc}")
        return None


def parse_optional_phonons(parser: CrystalOutputParser):
    try:
        freqs, all_freqs, intensities, degeneracies, irreps = parser.get_phonon_frequencies()

        # Store IR-active compact data for now.
        return {
            "frequencies_cm1": np.asarray(freqs, dtype=float),
            "intensities_km_mol": np.asarray(intensities, dtype=float),
            "degeneracies": np.asarray(degeneracies, dtype=int),
            "irreps": np.asarray(irreps),
            "all_frequencies_cm1_parser": np.asarray(all_freqs, dtype=float),
        }
    except Exception as exc:
        print(f"  warning: phonon parse failed: {exc}")
        return {
            "frequencies_cm1": None,
            "intensities_km_mol": None,
            "degeneracies": None,
            "irreps": None,
            "all_frequencies_cm1_parser": None,
        }


def build_one_reference(structure_dir: Path, ref_db_path: Path):
    structure = structure_dir.name
    freq_dir = structure_dir / "freq"

    print("-" * 72)
    print(structure)

    freq_out = find_single_outfile(freq_dir)
    hessfreq = find_hessfreq_file(freq_dir)

    if freq_out is None:
        print("  skipping: no freq .out file")
        return False

    if hessfreq is None:
        print("  skipping: no .hessfreq file")
        return False

    atoms, cif_source = load_reference_atoms(structure_dir, freq_out=freq_out)
    natoms = len(atoms)

    parser = CrystalOutputParser(freq_out)

    born, born_species = parse_optional_born(parser)
    dielectric = parse_optional_dielectric(parser)
    phonons = parse_optional_phonons(parser)

    print(f"  atoms: {natoms}")
    print(f"  freq_out: {freq_out.name}")
    print(f"  hessfreq: {hessfreq.name}")

    hessian_crystal = read_crystal_hessfreq_flat(hessfreq, n_atoms=natoms)
    hessian_cart_eV_A2 = hessian_crystal * HARTREE_PER_BOHR2_TO_EV_PER_A2

    hessian_mw_SI, eigvals_SI, eigvecs_mw, freqs_from_hessian_cm1, imag_flags = diagonalize_hessian(
        H_cart=hessian_cart_eV_A2,
        masses_amu=atoms.get_masses(),
        hessian_units="eV/Ang^2",
    )

    metadata = {
        "structure_name": structure,
        "source_dir": structure_dir,
        "freq_dir": freq_dir,
        "freq_out": freq_out,
        "hessfreq": hessfreq,
        "cif_source": cif_source,
        "hessian_original_units": "hartree/bohr^2",
        "hessian_stored_units": "eV/Ang^2",
        "frequencies_unit": "cm^-1",
        "intensities_unit": "KM/MOL",
        "database_role": "CRYSTAL reference",
    }

    write_crystal_reference(
        ref_db_path=ref_db_path,
        structure=structure,
        atoms=atoms,
        born_charges=born,
        born_species=born_species,
        dielectric_tensor=dielectric,
        hessian_cart_eV_A2=hessian_cart_eV_A2,
        hessian_mw_SI=hessian_mw_SI,
        eigvals_SI=eigvals_SI,
        eigvecs_mw=eigvecs_mw,
        frequencies_cm1=freqs_from_hessian_cm1,
        ir_frequencies_cm1=phonons["frequencies_cm1"],
        imag_flags=imag_flags,
        intensities_km_mol=phonons["intensities_km_mol"],
        degeneracies=phonons["degeneracies"],
        irreps=phonons["irreps"],
        metadata=metadata,
    )

    print("  written")
    return True


def build_ref_db(structures_dir: Path, ref_db_path: Path, only: list[str] | None = None):
    structures_dir = structures_dir.resolve()
    ref_db_path = ref_db_path.resolve()

    if not structures_dir.exists():
        raise FileNotFoundError(f"Missing structures directory: {structures_dir}")

    structure_dirs = sorted(p for p in structures_dir.iterdir() if p.is_dir())

    if only:
        allowed = set(only)
        structure_dirs = [p for p in structure_dirs if p.name in allowed]

    n_written = 0
    n_failed = 0

    for structure_dir in structure_dirs:
        try:
            ok = build_one_reference(structure_dir, ref_db_path)
            if ok:
                n_written += 1
        except Exception as exc:
            n_failed += 1
            print(f"  failed: {exc}")

    print("-" * 72)
    print(f"Done. written={n_written}, failed={n_failed}")
    print(f"DB: {ref_db_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build CRYSTAL reference DB.")
    parser.add_argument(
        "--structures-dir",
        type=Path,
        required=True,
        help="Directory containing <structure>/freq folders.",
    )
    parser.add_argument(
        "--ref-db",
        type=Path,
        default=ROOT / "data" / "ref_db.h5",
        help="Output reference HDF5 database.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional structure names to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    structures_dir = ROOT / args.structures_dir

    build_ref_db(
        structures_dir=structures_dir,
        ref_db_path=args.ref_db,
        only=args.only,
    )


if __name__ == "__main__":
    main()