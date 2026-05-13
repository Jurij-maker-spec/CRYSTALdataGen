from __future__ import annotations

import numpy as np
from ase import Atoms


PAIR_MIN_DIST = {
    ("O", "O"): 1.00,
    ("N", "N"): 0.95,
    ("C", "C"): 1.00,
    ("P", "P"): 1.80,
    ("Si", "Si"): 1.80,
    ("Ti", "Ti"): 2.00,
    ("Al", "Al"): 2.20,
    ("Na", "Na"): 2.20,
    ("Cu", "Cu"): 2.00,

    ("Ti", "O"): 1.40,
    ("Ti", "N"): 1.45,
    ("Ti", "C"): 1.45,
    ("Ti", "P"): 1.90,
    ("Al", "O"): 1.20,
    ("Al", "N"): 1.30,
    ("Na", "O"): 1.60,
    ("Cu", "O"): 1.40,
    ("Si", "O"): 1.35,
    ("Si", "N"): 1.40,
    ("Si", "P"): 1.70,
}

DEFAULT_MIN_DIST = 0.80


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def pair_min_dist(a: str, b: str) -> float:
    return PAIR_MIN_DIST.get(pair_key(a, b), DEFAULT_MIN_DIST)


def minimum_image_shortest_contact(atoms: Atoms) -> dict:
    cell = np.asarray(atoms.cell.array)
    frac = atoms.get_scaled_positions(wrap=True)
    syms = atoms.get_chemical_symbols()

    dmin = np.inf
    best = None

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            df = frac[j] - frac[i]
            df -= np.round(df)

            dr = df @ cell
            d = float(np.linalg.norm(dr))
            allowed = pair_min_dist(syms[i], syms[j])

            if d < dmin:
                dmin = d
                best = {
                    "shortest_contact_A": dmin,
                    "shortest_contact_pair": [i + 1, j + 1],
                    "shortest_contact_symbols": [syms[i], syms[j]],
                    "shortest_contact_threshold_A": allowed,
                    "shortest_contact_frac_delta": df.tolist(),
                    "shortest_contact_cart_delta_A": dr.tolist(),
                    "has_short_contact": dmin < allowed,
                }

    if best is None:
        raise ValueError("Cannot check contacts for structure with fewer than 2 atoms")

    return best


def assert_structure_is_reasonable(atoms: Atoms, label: str) -> None:
    info = minimum_image_shortest_contact(atoms)

    if info["has_short_contact"]:
        raise ValueError(
            f"[{label}] too short contact: "
            f"{info['shortest_contact_symbols']} "
            f"{info['shortest_contact_pair']} = "
            f"{info['shortest_contact_A']:.4f} Å "
            f"< {info['shortest_contact_threshold_A']:.4f} Å"
        )
    