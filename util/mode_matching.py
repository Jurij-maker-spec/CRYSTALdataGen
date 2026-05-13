#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from ase import Atoms
import re
import spglib
import matplotlib as mpl
from .phonons import diagonalize_hessian
from .ref_db import read_crystal_modes

mpl.rcParams['font.size'] = 14


def fix_label_case(label):
    """Convert an atomic label like FE1 -> Fe1 or SI2 -> Si."""
    match = re.match(r"([A-Za-z]+)([0-9]*)", label)
    if not match:
        return label  # leave unchanged if not matching expected pattern
    elem, idx = match.groups()
    # Only capitalize first letter, lowercase the rest (Fe, Si, Co, etc.)
    elem_fixed = elem.capitalize()
    return elem_fixed


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


def reorder_hessian(H, perm):
    """
    Reorder Hessian so that new order follows the reference atom order.
    """
    idx = []
    for a in perm:
        idx.extend([3*a, 3*a + 1, 3*a + 2])
    idx = np.array(idx, dtype=int)
    H_new = H[np.ix_(idx, idx)]
    return 0.5 * (H_new + H_new.T)


def fractional_pbc_diff(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return d - np.round(d)


def build_atom_permutation_by_fracpos(ref_atoms, test_atoms, tol=1e-3):
    ref_syms = ref_atoms.get_chemical_symbols()
    test_syms = test_atoms.get_chemical_symbols()

    ref_frac = ref_atoms.get_scaled_positions(wrap=True)
    test_frac = test_atoms.get_scaled_positions(wrap=True)

    n = len(ref_atoms)
    used = np.zeros(n, dtype=bool)
    perm = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = []
        for j in range(n):
            if used[j]:
                continue
            if ref_syms[i] != test_syms[j]:
                continue
            dist = np.linalg.norm(fractional_pbc_diff(ref_frac[i], test_frac[j]))
            if dist < tol:
                candidates.append((j, dist))

        if not candidates:
            raise ValueError(f"No match found for atom {i} ({ref_syms[i]})")

        candidates.sort(key=lambda x: x[1])
        perm[i] = candidates[0][0]
        used[perm[i]] = True

    return perm


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
    atoms = standardize_to_primitive(atoms, no_idealize=False)
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


# ============================================================
# Mode comparison helpers
# ============================================================

def normalize_mode_columns(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    norms = np.linalg.norm(U, axis=0)
    norms[norms == 0.0] = 1.0
    return U / norms[None, :]


def compute_overlap_matrix(U_ref: np.ndarray, U_test: np.ndarray, absolute: bool = True) -> np.ndarray:
    """
    Mode-overlap matrix between two mode sets.
    Columns of U_ref/U_test are modes.
    """
    A = normalize_mode_columns(U_ref)
    B = normalize_mode_columns(U_test)
    O = A.T @ B
    return np.abs(O) if absolute else O


def match_modes_hungarian(
    freqs_ref: np.ndarray,
    U_ref: np.ndarray,
    freqs_test: np.ndarray,
    U_test: np.ndarray,
    skip_first: int = 3,
):
    """
    Global one-to-one mode matching by overlap maximization.
    """
    O = compute_overlap_matrix(U_ref[:, skip_first:], U_test[:, skip_first:], absolute=True)
    cost = 1.0 - O
    row_ind, col_ind = linear_sum_assignment(cost)

    rows = row_ind + skip_first
    cols = col_ind + skip_first
    overlaps = O[row_ind, col_ind]

    matches = []
    for i_ref, i_test, ov in sorted(zip(rows, cols, overlaps), key=lambda x: x[0]):
        matches.append(
            {
                "mode_ref": int(i_ref),
                "mode_test": int(i_test),
                "freq_ref": float(freqs_ref[i_ref]),
                "freq_test": float(freqs_test[i_test]),
                "delta_cm1": float(freqs_test[i_test] - freqs_ref[i_ref]),
                "overlap": float(ov),
            }
        )
    return matches


def group_degenerate_modes(freqs_cm: np.ndarray, tol_cm1: float = 1.0):
    """
    Group adjacent modes whose frequencies differ by <= tol_cm1.
    """
    freqs_cm = np.asarray(freqs_cm, dtype=float)
    groups = []
    start = 0
    for i in range(1, len(freqs_cm)):
        if abs(freqs_cm[i] - freqs_cm[i - 1]) > tol_cm1:
            groups.append(np.arange(start, i))
            start = i
    groups.append(np.arange(start, len(freqs_cm)))
    return groups


def subspace_overlap(U_ref_block: np.ndarray, U_test_block: np.ndarray) -> float:
    """
    Compare two mode subspaces. 1.0 means identical subspaces.
    """
    O = normalize_mode_columns(U_ref_block).T @ normalize_mode_columns(U_test_block)
    return np.linalg.norm(O, ord="fro") / np.sqrt(min(O.shape))


def match_degenerate_groups(
    freqs_ref: np.ndarray,
    U_ref: np.ndarray,
    freqs_test: np.ndarray,
    U_test: np.ndarray,
    skip_first: int = 3,
    degeneracy_tol: float = 1.0,
    freq_weight: float = 0.02,
):
    """
    Match degenerate/near-degenerate mode groups between reference and test.

    Cost combines:
      - subspace mismatch: 1 - subspace_overlap
      - group-center frequency mismatch

    Parameters
    ----------
    freq_weight : float
        Weight for frequency mismatch term in the cost.
        Cost += freq_weight * abs(mean(freq_ref_group) - mean(freq_test_group))

    Returns
    -------
    ref_groups_full : list[np.ndarray]
    test_groups_full : list[np.ndarray]
    group_matches : list[dict]
    group_overlap_matrix : ndarray
    """
    ref_groups = group_degenerate_modes(freqs_ref[skip_first:], tol_cm1=degeneracy_tol)
    test_groups = group_degenerate_modes(freqs_test[skip_first:], tol_cm1=degeneracy_tol)

    ref_groups_full = [g + skip_first for g in ref_groups]
    test_groups_full = [g + skip_first for g in test_groups]

    nref = len(ref_groups_full)
    ntest = len(test_groups_full)

    overlap_mat = np.zeros((nref, ntest), dtype=float)
    cost_mat = np.full((nref, ntest), 1.0e6, dtype=float)

    for i, gref in enumerate(ref_groups_full):
        Uref = U_ref[:, gref]
        fref = np.mean(freqs_ref[gref])

        for j, gtest in enumerate(test_groups_full):
            Utest = U_test[:, gtest]
            ftest = np.mean(freqs_test[gtest])

            # Prefer equal group sizes, but still allow mismatch if needed
            size_penalty = 0.0 if len(gref) == len(gtest) else 0.25 * abs(len(gref) - len(gtest))

            ov = subspace_overlap(Uref, Utest)
            overlap_mat[i, j] = ov

            cost_mat[i, j] = (
                (1.0 - ov)
                + freq_weight * abs(fref - ftest)
                + size_penalty
            )

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    group_matches = []
    for i, j in sorted(zip(row_ind, col_ind), key=lambda x: x[0]):
        gref = ref_groups_full[i]
        gtest = test_groups_full[j]
        group_matches.append(
            {
                "group_ref_index": int(i),
                "group_test_index": int(j),
                "ref_modes": gref,
                "test_modes": gtest,
                "ref_freq_mean": float(np.mean(freqs_ref[gref])),
                "test_freq_mean": float(np.mean(freqs_test[gtest])),
                "delta_cm1": float(np.mean(freqs_test[gtest]) - np.mean(freqs_ref[gref])),
                "subspace_overlap": float(overlap_mat[i, j]),
                "size_ref": int(len(gref)),
                "size_test": int(len(gtest)),
            }
        )

    return ref_groups_full, test_groups_full, group_matches, overlap_mat


def plot_mode_overlap_heatmap(
    overlap_matrix: np.ndarray,
    freqs_ref: np.ndarray | None = None,
    freqs_test: np.ndarray | None = None,
    skip_first: int = 3,
    outfile: str | Path = "mode_overlap_heatmap.png",
    title: str = "Mode overlap",
):
    """
    Heatmap of |<u_ref|u_test>| after skipping acoustic modes.
    """

    O = np.asarray(overlap_matrix, dtype=float)
    outfile = Path(outfile)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(O, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$| \langle u_\mathrm{ref} | u_\mathrm{test} \rangle |$")

    ax.set_xlabel("Test modes")
    ax.set_ylabel("Reference modes")
    ax.set_title(title)

    if freqs_ref is not None:
        ylabels = [f"{f:.0f}" for f in freqs_ref[skip_first:]]
        step_y = max(1, len(ylabels) // 12)
        ax.set_yticks(np.arange(0, len(ylabels), step_y))
        ax.set_yticklabels(ylabels[::step_y])

    if freqs_test is not None:
        xlabels = [f"{f:.0f}" for f in freqs_test[skip_first:]]
        step_x = max(1, len(xlabels) // 12)
        ax.set_xticks(np.arange(0, len(xlabels), step_x))
        ax.set_xticklabels(xlabels[::step_x], rotation=90)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_group_overlap_heatmap(
    group_overlap_matrix: np.ndarray,
    ref_groups: list,
    test_groups: list,
    group_matches: list,
    freqs_ref: np.ndarray,
    freqs_test: np.ndarray,
    outfile: str | Path = "group_overlap_heatmap.png",
    title: str = "Degenerate-group overlap",
):
    """
    Heatmap of subspace overlaps between degenerate/near-degenerate groups.
    """
    O = np.asarray(group_overlap_matrix, dtype=float)
    outfile = Path(outfile)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(O, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Subspace overlap", size=14)


    def group_label_cm(groups, freqs):
        labels = []
        for g in groups:
            if len(g) == 1:
                labels.append(f"{freqs[g[0]]:.0f}")
            else:
                labels.append(f"{freqs[g[0]]:.0f}-{freqs[g[-1]]:.0f}")
        return labels
    


    def group_label_modes(groups):
        xlabels = []
        ylabels = []
        for g in groups:
            x = g['test_modes']
            y = g['ref_modes']
            if len(x) == 1:
                xlabels.append(f"{x[0]}")
            else:
                xlabels.append(f"{x[0]}-{x[-1]}")
            if len(y) == 1:
                ylabels.append(f"{y[0]}")
            else:
                ylabels.append(f"{y[0]}-{y[-1]}")

        return xlabels, ylabels


    #ylabels = group_label_cm(ref_groups, freqs_ref)
    #xlabels = group_label_cm(test_groups, freqs_test)

    xlabels, ylabels = group_label_modes(group_matches)
    


    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90)

    ax.set_xlabel("Test groups, cm⁻¹", size=12)
    ax.set_ylabel("Reference groups, cm⁻¹", size=12)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def print_group_match_summary(group_matches):
    print("\nDegenerate-group matching summary:")
    print(
        f"{'g_ref':>6s} {'g_test':>6s} "
        f"{'ref_modes':>16s} {'test_modes':>16s} "
        f"{'dfreq':>10s} {'subspace':>10s}"
    )
    print("-" * 74)

    for g in group_matches:
        ref_modes = [int(x + 1) for x in g["ref_modes"]]
        test_modes = [int(x + 1) for x in g["test_modes"]]
        print(
            f"{g['group_ref_index']+1:6d} {g['group_test_index']+1:6d} "
            f"{str(ref_modes):>16s} {str(test_modes):>16s} "
            f"{g['delta_cm1']:10.4f} {g['subspace_overlap']:10.4f}"
        )


def print_mode_match_summary(matches):
    print("\nMode matching summary:")
    print(
        f"{'ref':>5s} {'test':>5s} "
        f"{'freq_ref':>12s} {'freq_test':>12s} "
        f"{'delta':>12s} {'|overlap|':>12s}"
    )
    print("-" * 62)
    for m in matches:
        print(
            f"{m['mode_ref']+1:5d} {m['mode_test']+1:5d} "
            f"{m['freq_ref']:12.4f} {m['freq_test']:12.4f} "
            f"{m['delta_cm1']:12.4f} {m['overlap']:12.4f}"
        )


def run_mode_comparison(
    atoms: Atoms,
    mace_modes: dict,
    crystal_hess_path: str | Path,
    freq_out_path: str | Path,
    crystal_hessian_units: str = "hartree/bohr^2",
    skip_first: int = 3,
    degeneracy_tol: float = 1.0,
    heatmap_outfile: str | Path | None = None,
    title: str = "CRYSTAL vs MACE mode overlap",
):
    """
    Full comparison workflow:
    - read CRYSTAL Hessian
    - diagonalize with the same convention as MACE
    - compute overlap matrix
    - match modes
    - optionally save heatmap
    """
    n_atoms = len(atoms)
    masses_amu = atoms.get_masses()

    H_crys = read_crystal_hessfreq_flat(crystal_hess_path, n_atoms=n_atoms)

    crystal_atoms = read_crystal_primitive_atoms_from_freq_out(freq_out_file=freq_out_path)

    perm = build_atom_permutation_by_fracpos(
        ref_atoms=atoms,
        test_atoms=crystal_atoms,
        tol=1e-1,
    )

    H_crys = reorder_hessian(H_crys, perm)

    Hmw_crys, eigvals_crys, evecs_crys, freqs_crys, imag_crys = diagonalize_hessian(
        H_crys,
        masses_amu,
        hessian_units=crystal_hessian_units,
    )

    freqs_mace = mace_modes["freqs_cm"]
    evecs_mace = mace_modes["eigvecs_mw"]

    overlap_full = compute_overlap_matrix(evecs_crys, evecs_mace, absolute=True)
    overlap_cut = overlap_full[skip_first:, skip_first:]

    matches = match_modes_hungarian(
        freqs_ref=freqs_crys,
        U_ref=evecs_crys,
        freqs_test=freqs_mace,
        U_test=evecs_mace,
        skip_first=skip_first,
    )

    ref_groups = group_degenerate_modes(freqs_crys[skip_first:], tol_cm1=degeneracy_tol)
    test_groups = group_degenerate_modes(freqs_mace[skip_first:], tol_cm1=degeneracy_tol)

    ref_groups, test_groups, group_matches, group_overlap_matrix = match_degenerate_groups(
        freqs_ref=freqs_crys,
        U_ref=evecs_crys,
        freqs_test=freqs_mace,
        U_test=evecs_mace,
        skip_first=skip_first,
        degeneracy_tol=degeneracy_tol,
        freq_weight=0.02,
    )

    subgroup_results = group_matches

    group_heatmap_outfile = None
    if heatmap_outfile is not None:
        heatmap_outfile = Path(heatmap_outfile)

        plot_mode_overlap_heatmap(
            overlap_cut,
            freqs_ref=freqs_crys,
            freqs_test=freqs_mace,
            skip_first=skip_first,
            outfile=heatmap_outfile,
            title=title,
        )

        group_heatmap_outfile = heatmap_outfile.with_name(
            heatmap_outfile.stem + "_groups" + heatmap_outfile.suffix
        )

        plot_group_overlap_heatmap(
            group_overlap_matrix=group_overlap_matrix,
            ref_groups=ref_groups,
            test_groups=test_groups,
            group_matches=group_matches,
            freqs_ref=freqs_crys,
            freqs_test=freqs_mace,
            outfile=group_heatmap_outfile,
            title=title + " (degenerate groups)",
        )


    return {
        "crystal": {
            "H_cart": H_crys,
            "H_mw": Hmw_crys,
            "eigvals_SI": eigvals_crys,
            "eigvecs_mw": evecs_crys,
            "freqs_cm": freqs_crys,
            "imag_flags": imag_crys,
        },
        "mace": mace_modes,
        "overlap_full": overlap_full,
        "overlap_cut": overlap_cut,
        "matches": matches,
        "subgroups": subgroup_results,
        "ref_groups": ref_groups,
        "test_groups": test_groups,
        "group_overlap_matrix": group_overlap_matrix,
        "group_heatmap_outfile": group_heatmap_outfile,
        "skip_first": skip_first,
        "degeneracy_tol": degeneracy_tol,
    }


def run_mode_comparison_from_ref_db(
    ref_db_path: str | Path,
    structure: str,
    mace_modes: dict,
    skip_first: int = 3,
    degeneracy_tol: float = 1.0,
    heatmap_outfile: str | Path | None = None,
    title: str | None = None,
):
    """
    Mode comparison using CRYSTAL mode data already stored in ref_db.h5.

    This avoids reparsing:
      - *_freq.out
      - *.hessfreq

    It assumes that build_ref_db.py stored CRYSTAL eigvecs_mw and frequencies_cm1
    using the same atom order convention as the reference DB geometry.
    """
    crystal = read_crystal_modes(ref_db_path, structure)

    freqs_crys = np.asarray(crystal["freqs_cm"], dtype=float)
    evecs_crys = np.asarray(crystal["eigvecs_mw"], dtype=float)

    freqs_mace = np.asarray(mace_modes["freqs_cm"], dtype=float)
    evecs_mace = np.asarray(mace_modes["eigvecs_mw"], dtype=float)

    if evecs_crys.shape != evecs_mace.shape:
        raise ValueError(
            f"Mode eigenvector shape mismatch for {structure}: "
            f"CRYSTAL {evecs_crys.shape}, MACE {evecs_mace.shape}. "
            "This usually means atom count/order or primitive-cell convention differs."
        )

    overlap_full = compute_overlap_matrix(evecs_crys, evecs_mace, absolute=True)
    overlap_cut = overlap_full[skip_first:, skip_first:]

    matches = match_modes_hungarian(
        freqs_ref=freqs_crys,
        U_ref=evecs_crys,
        freqs_test=freqs_mace,
        U_test=evecs_mace,
        skip_first=skip_first,
    )

    ref_groups, test_groups, group_matches, group_overlap_matrix = match_degenerate_groups(
        freqs_ref=freqs_crys,
        U_ref=evecs_crys,
        freqs_test=freqs_mace,
        U_test=evecs_mace,
        skip_first=skip_first,
        degeneracy_tol=degeneracy_tol,
        freq_weight=0.02,
    )

    group_heatmap_outfile = None

    if heatmap_outfile is not None:
        heatmap_outfile = Path(heatmap_outfile)

        if title is None:
            title = f"{structure}: CRYSTAL DB vs MACE mode overlap"

        plot_mode_overlap_heatmap(
            overlap_cut,
            freqs_ref=freqs_crys,
            freqs_test=freqs_mace,
            skip_first=skip_first,
            outfile=heatmap_outfile,
            title=title,
        )

        group_heatmap_outfile = heatmap_outfile.with_name(
            heatmap_outfile.stem + "_groups" + heatmap_outfile.suffix
        )

        plot_group_overlap_heatmap(
            group_overlap_matrix=group_overlap_matrix,
            ref_groups=ref_groups,
            test_groups=test_groups,
            group_matches=group_matches,
            freqs_ref=freqs_crys,
            freqs_test=freqs_mace,
            outfile=group_heatmap_outfile,
            title=title + " (degenerate groups)",
        )

    return {
        "crystal": crystal,
        "mace": mace_modes,
        "overlap_full": overlap_full,
        "overlap_cut": overlap_cut,
        "matches": matches,
        "subgroups": group_matches,
        "ref_groups": ref_groups,
        "test_groups": test_groups,
        "group_overlap_matrix": group_overlap_matrix,
        "group_heatmap_outfile": group_heatmap_outfile,
        "skip_first": skip_first,
        "degeneracy_tol": degeneracy_tol,
        "source": "ref_db",
    }
