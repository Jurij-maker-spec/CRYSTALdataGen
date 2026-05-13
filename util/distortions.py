from __future__ import annotations

from typing import Any
from pathlib import Path
import numpy as np
from ase import Atoms

from .structure_io import wrap_atoms
from .distance_checks import minimum_image_shortest_contact


def random_symmetric_strain(rng: np.random.Generator, max_abs: float) -> np.ndarray:
    eps = rng.uniform(-max_abs, max_abs, size=(3, 3))
    eps = 0.5 * (eps + eps.T)
    return np.eye(3) + eps


def distort_total_random(atoms: Atoms, cfg: dict[str, Any]) -> tuple[Atoms, dict]:
    """
    Reproduces generate_random_displ.py logic.

    Important:
    - global np.random state
    - uniform Cartesian displacements
    - full non-symmetric random 3x3 strain
    - displacement first, strain second
    """

    distorted = atoms.copy()
    old_cell = distorted.cell.array.copy()

    disp = np.random.uniform(
        -cfg["DISP_MAX"],
        cfg["DISP_MAX"],
        distorted.positions.shape,
    )
    distorted.positions += disp

    F = np.eye(3) + np.random.uniform(
        -cfg["STRAIN_MAX"],
        cfg["STRAIN_MAX"],
        (3, 3),
    )

    distorted.set_cell(distorted.cell @ F, scale_atoms=True)
    distorted = wrap_atoms(distorted)

    disp_norms = np.linalg.norm(disp, axis=1)

    meta = {
        "distortion_mode": "total_random",
        "mode": "total_random",
        "disp_rms_A": float(np.sqrt(np.mean(np.sum(disp**2, axis=1)))),
        "disp_mean_A": float(np.mean(disp_norms)),
        "disp_max_norm_A": float(np.max(disp_norms)),
        "max_disp_A": cfg["DISP_MAX"],
        "strain_matrix": F.tolist(),
        "strain_norm": float(np.linalg.norm(F - np.eye(3))),
        "max_strain": cfg["STRAIN_MAX"],
        "strain_is_symmetric": False,
        "old_cell_A": old_cell.tolist(),
        "new_cell_A": distorted.cell.array.tolist(),
    }

    return distorted, meta


def distort_mixed_random(
    atoms: Atoms,
    rng: np.random.Generator,
    cfg: dict[str, Any],
) -> tuple[Atoms, dict]:
    distorted = atoms.copy()

    disp = rng.uniform(
        -cfg["DISP_MAX"],
        cfg["DISP_MAX"],
        size=distorted.positions.shape,
    )
    distorted.positions += disp

    F = random_symmetric_strain(rng, cfg["STRAIN_MAX"])
    distorted.set_cell(distorted.cell @ F, scale_atoms=True)
    distorted = wrap_atoms(distorted)

    disp_norms = np.linalg.norm(disp, axis=1)

    meta = {
        "distortion_mode": "mixed_random",
        "mode": "mixed_random",
        "disp_rms_A": float(np.sqrt(np.mean(np.sum(disp**2, axis=1)))),
        "disp_mean_A": float(np.mean(disp_norms)),
        "disp_max_norm_A": float(np.max(disp_norms)),
        "max_disp_A": cfg["DISP_MAX"],
        "strain_matrix": F.tolist(),
        "strain_norm": float(np.linalg.norm(F - np.eye(3))),
        "max_strain": cfg["STRAIN_MAX"],
        "strain_is_symmetric": True,
    }

    return distorted, meta


def build_mode_schedule(n_structures: int, cfg: dict[str, Any]) -> list[str]:
    n_small = int(round(cfg["FRAC_SMALL_DISP"] * n_structures))
    n_medium = int(round(cfg["FRAC_MEDIUM_DISP"] * n_structures))
    n_strain = int(round(cfg["FRAC_STRAIN_ONLY"] * n_structures))

    n_used = n_small + n_medium + n_strain
    n_mixed = n_structures - n_used

    return (
        ["small_disp"] * n_small
        + ["medium_disp"] * n_medium
        + ["strain_only"] * n_strain
        + ["mixed"] * n_mixed
    )


def distort_scheduled_random(
    atoms: Atoms,
    mode: str,
    rng: np.random.Generator,
    cfg: dict[str, Any],
) -> tuple[Atoms, dict]:
    distorted = atoms.copy()
    disp = np.zeros_like(distorted.positions)
    F = np.eye(3)

    if mode == "small_disp":
        disp = rng.uniform(-cfg["SMALL_DISP_MAX"], cfg["SMALL_DISP_MAX"], distorted.positions.shape)
        distorted.positions += disp

    elif mode == "medium_disp":
        disp = rng.uniform(-cfg["MEDIUM_DISP_MAX"], cfg["MEDIUM_DISP_MAX"], distorted.positions.shape)
        distorted.positions += disp

    elif mode == "strain_only":
        F = random_symmetric_strain(rng, cfg["SMALL_STRAIN_MAX"])
        distorted.set_cell(distorted.cell @ F, scale_atoms=True)

    elif mode == "mixed":
        disp = rng.uniform(-cfg["SMALL_DISP_MAX"], cfg["SMALL_DISP_MAX"], distorted.positions.shape)
        distorted.positions += disp
        F = random_symmetric_strain(rng, cfg["MEDIUM_STRAIN_MAX"])
        distorted.set_cell(distorted.cell @ F, scale_atoms=True)

    else:
        raise ValueError(f"Unknown scheduled mode: {mode}")

    distorted = wrap_atoms(distorted)
    disp_norms = np.linalg.norm(disp, axis=1)

    meta = {
        "distortion_mode": "scheduled_random",
        "mode": mode,
        "disp_rms_A": float(np.sqrt(np.mean(np.sum(disp**2, axis=1)))),
        "disp_mean_A": float(np.mean(disp_norms)),
        "disp_max_norm_A": float(np.max(disp_norms)),
        "strain_matrix": F.tolist(),
        "strain_norm": float(np.linalg.norm(F - np.eye(3))),
        "strain_is_symmetric": True,
    }

    return distorted, meta


def generate_valid_distortion(
    atoms: Atoms,
    cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
    scheduled_mode: str | None = None,
) -> tuple[Atoms, dict]:
    last_fail = None

    for attempt in range(1, cfg["MAX_GENERATION_ATTEMPTS_PER_STRUCTURE"] + 1):
        mode = cfg["DISTORTION_MODE"]

        if mode == "total_random":
            distorted, meta = distort_total_random(atoms, cfg)

        elif mode == "mixed_random":
            if rng is None:
                raise ValueError("mixed_random requires rng")
            distorted, meta = distort_mixed_random(atoms, rng, cfg)

        elif mode == "scheduled_random":
            if rng is None or scheduled_mode is None:
                raise ValueError("scheduled_random requires rng and scheduled_mode")
            distorted, meta = distort_scheduled_random(atoms, scheduled_mode, rng, cfg)

        else:
            raise ValueError(f"Unsupported random distortion mode here: {mode}")

        contact = minimum_image_shortest_contact(distorted)

        meta.update(contact)
        meta["generation_attempt"] = attempt

        if not contact["has_short_contact"] or not cfg["REJECT_ON_SHORT_DISTANCE"]:
            return distorted, meta

        last_fail = contact

    raise RuntimeError(
        f"Could not generate valid distortion after "
        f"{cfg['MAX_GENERATION_ATTEMPTS_PER_STRUCTURE']} attempts. "
        f"Last failure: {last_fail}"
    )


def check_template_compatible(atoms: Atoms, template_data: dict[str, np.ndarray]) -> None:
    atom_symbols = atoms.get_chemical_symbols()
    template_symbols = [str(x) for x in template_data["symbols"]]

    if atom_symbols != template_symbols:
        raise ValueError(
            "Template symbols/order do not match current reference atoms.\n"
            f"Current:  {atom_symbols}\n"
            f"Template: {template_symbols}"
        )

    if len(atoms) != len(template_symbols):
        raise ValueError(
            f"Template atom count mismatch: atoms={len(atoms)}, template={len(template_symbols)}"
        )


def load_template_data(template_file: Path) -> dict[str, np.ndarray]:
    template_file = Path(template_file)

    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    data = np.load(template_file, allow_pickle=True)

    required = [
        "ids",
        "frac_disp",
        "cell_deformation",
        "symbols",
    ]

    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Template file missing required arrays: {missing}")

    return {k: data[k] for k in data.files}


def apply_template_distortion(
    atoms: Atoms,
    template_data: dict,
    template_index: int,
    cfg: dict[str, Any],
):
    ids = template_data["ids"]
    frac_disp = template_data["frac_disp"][template_index]

    distorted = atoms.copy()

    old_cell = distorted.cell.array
    new_cell = old_cell.copy()

    if cfg["TEMPLATE_USE_CELL_DEFORMATION"]:
        F = template_data["cell_deformation"][template_index]
        scale = cfg["TEMPLATE_CELL_SCALE"]
        F_scaled = np.eye(3) + scale * (F - np.eye(3))
        new_cell = old_cell @ F_scaled
        distorted.set_cell(new_cell, scale_atoms=False)

    if cfg["TEMPLATE_USE_FRAC_DISP"]:
        atomic_scale = cfg["TEMPLATE_ATOMIC_SCALE"]
        new_scaled = atoms.get_scaled_positions(wrap=False) + atomic_scale * frac_disp
        distorted.set_scaled_positions(new_scaled)

    distorted.set_pbc(True)
    distorted.wrap()

    norms = np.linalg.norm((cfg["TEMPLATE_ATOMIC_SCALE"] * frac_disp) @ new_cell, axis=1)

    meta = {
        "mode": "reuse_template",
        "template_id": str(ids[template_index]),
        "template_index": int(template_index),
        "template_use_cell_deformation": cfg["TEMPLATE_USE_CELL_DEFORMATION"],
        "template_use_frac_disp": cfg["TEMPLATE_USE_FRAC_DISP"],
        "template_cell_scale": cfg["TEMPLATE_CELL_SCALE"],
        "template_atomic_scale": cfg["TEMPLATE_ATOMIC_SCALE"],
        "disp_rms_A": float(np.sqrt(np.mean(norms**2))),
        "max_disp_A": float(np.max(norms)),
        "strain_norm": float(np.linalg.norm(new_cell - old_cell)),
    }

    return distorted, meta
