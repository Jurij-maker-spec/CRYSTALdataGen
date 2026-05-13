from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from ase import Atoms
from ase.io import read
from ase.build import make_supercell
import numpy as np


def wrap_atoms(atoms: Atoms) -> Atoms:
    out = atoms.copy()
    out.set_pbc(True)
    out.wrap()
    return out


def report_structure(atoms: Atoms, label: str) -> None:
    counts = Counter(atoms.get_chemical_symbols())
    print(f"[{label}] natoms  = {len(atoms)}")
    print(f"[{label}] formula = {atoms.get_chemical_formula()}")
    print(f"[{label}] counts  = {dict(counts)}")


def find_structure_cif(structure_dir: Path, structure_name: str, cfg: dict) -> Path:
    if cfg.get("CIF_FILE") is not None:
        return Path(cfg["CIF_FILE"])

    candidates = [
        structure_dir / f"{structure_name}_geoopt_clean.cif",
    ]

    existing = [p for p in candidates if p.exists()]

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple CIF candidates found for {structure_name}: "
            + ", ".join(str(p) for p in existing)
        )

    raise FileNotFoundError(
        f"No CIF found for {structure_name}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def load_reference_atoms(structure_dir: Path, cfg: dict) -> tuple[Atoms, Path]:
    structure_name = structure_dir.name
    cif_file = find_structure_cif(structure_dir, structure_name, cfg)
    atoms = read(cif_file)
    atoms.set_pbc(True)
    return atoms, cif_file


def apply_supercell_if_requested(atoms: Atoms, cfg: dict) -> tuple[Atoms, dict]:
    if cfg["SUPERCELL"] is None:
        return wrap_atoms(atoms), {
            "supercell": None,
            "supercell_stage": None,
        }

    sc = list(cfg["SUPERCELL"])
    scmat = np.diag(sc)

    out = make_supercell(atoms, scmat)
    out = wrap_atoms(out)

    return out, {
        "supercell": sc,
        "supercell_stage": cfg.get("SUPERCELL_STAGE", "before_distortion"),
    }


def fix_label_case(label: str) -> str:
    match = re.match(r"([A-Za-z]+)([0-9]*)", label)
    if not match:
        return label
    elem, idx = match.groups()
    return elem.capitalize() + idx

