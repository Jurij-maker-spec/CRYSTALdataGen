#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from ase import Atoms
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from .phonons import diagonalize_hessian
from .ref_db import (
    read_crystal_modes,
    read_crystal_hessfreq_flat,
    read_crystal_primitive_atoms_from_freq_out
)
mpl.style.use('/home/jha/jha/python_scripts/CRYSTALdataGen/util/style.mplstyle')
CMAP = 'managua_r'

def _slice_CMAP(cmap):
    full_cmap = cm.get_cmap(cmap)
    colors = full_cmap(np.linspace(0.4, 1.0, 512))
    sliced_cmap = ListedColormap(colors)
    return sliced_cmap

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


def reorder_mode_eigenvectors(U: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Reorder mode eigenvectors by atom permutation.

    Parameters
    ----------
    U
        Eigenvectors with shape (3N, Nmodes).
    perm
        Atom indices in the old order, arranged into the new order.

    Returns
    -------
    U_new
        Eigenvectors reordered to the new atom order.
    """
    idx = []
    for a in perm:
        idx.extend([3 * a, 3 * a + 1, 3 * a + 2])
    idx = np.asarray(idx, dtype=int)
    return np.asarray(U)[idx, :]


def fractional_pbc_diff(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return d - np.round(d)


def build_atom_permutation_by_fracpos(ref_atoms, test_atoms, tol=1e-2):
    """
    Match atoms by species and fractional coordinates, allowing a global
    origin shift.

    Returns
    -------
    perm_ref_in_test_order : ndarray
        Indices into ref_atoms ordered like test_atoms.

        If H_ref is in ref_atoms order, then

            H_ref_test_order = reorder_hessian(H_ref, perm_ref_in_test_order)

        gives the Hessian in test_atoms/MACE order.

    origin_shift : ndarray
        Fractional shift applied to test atoms before matching.
    max_mismatch : float
        Largest matched fractional minimum-image distance.
    """
    ref_syms = np.array(ref_atoms.get_chemical_symbols())
    test_syms = np.array(test_atoms.get_chemical_symbols())

    ref_frac = ref_atoms.get_scaled_positions(wrap=True)
    test_frac = test_atoms.get_scaled_positions(wrap=True)

    if len(ref_atoms) != len(test_atoms):
        raise ValueError(
            f"Atom count mismatch: ref={len(ref_atoms)}, test={len(test_atoms)}"
        )

    if sorted(ref_syms) != sorted(test_syms):
        raise ValueError(
            f"Species mismatch:\nref={list(ref_syms)}\ntest={list(test_syms)}"
        )

    def score_shift(shift):
        shifted_test_frac = (test_frac + shift) % 1.0

        total_cost = 0.0
        max_dist = 0.0
        perm_ref_in_test_order = np.full(len(ref_atoms), -1, dtype=int)

        for sym in sorted(set(ref_syms)):
            ref_idx = np.where(ref_syms == sym)[0]
            test_idx = np.where(test_syms == sym)[0]

            cost = np.zeros((len(test_idx), len(ref_idx)), dtype=float)

            for a, i_test in enumerate(test_idx):
                for b, i_ref in enumerate(ref_idx):
                    d = fractional_pbc_diff(
                        shifted_test_frac[i_test],
                        ref_frac[i_ref],
                    )
                    cost[a, b] = np.linalg.norm(d)

            row_ind, col_ind = linear_sum_assignment(cost)
            matched = cost[row_ind, col_ind]

            total_cost += float(np.sum(matched))
            max_dist = max(max_dist, float(np.max(matched)))

            for row, col in zip(row_ind, col_ind):
                i_test = test_idx[row]
                i_ref = ref_idx[col]
                perm_ref_in_test_order[i_test] = i_ref

        return total_cost, max_dist, perm_ref_in_test_order

    # Candidate origin shifts from same-species atom pairs.
    candidate_shifts = []

    for i_ref, sym_ref in enumerate(ref_syms):
        for i_test, sym_test in enumerate(test_syms):
            if sym_ref != sym_test:
                continue

            shift = ref_frac[i_ref] - test_frac[i_test]
            shift = shift - np.floor(shift)
            candidate_shifts.append(shift)

    # Also test no shift explicitly.
    candidate_shifts.append(np.zeros(3))

    best = None

    for shift in candidate_shifts:
        total_cost, max_dist, perm = score_shift(shift)

        if best is None or total_cost < best[0]:
            best = (total_cost, max_dist, perm, shift)

    total_cost, max_dist, perm_ref_in_test_order, origin_shift = best

    if np.any(perm_ref_in_test_order < 0):
        raise ValueError("Internal atom matching error: incomplete permutation.")

    if max_dist > tol:
        print(
            "Atom matching failed after origin-shift search. "
            f"\nBest shift={origin_shift}, max fractional mismatch={max_dist:.6g}, "
            f"\ntol={tol}"
            "\nTrying looser tolerance"
            )
        succes = False
        return perm_ref_in_test_order, origin_shift, max_dist, succes

        # raise ValueError(
        #     "Atom matching failed after origin-shift search. "
        #     f"Best shift={origin_shift}, max fractional mismatch={max_dist:.6g}, "
        #     f"tol={tol}"
        # )
    succes = True
    print("Atom matching successful.")
    print(f"  origin shift applied to test atoms: {origin_shift}")
    print(f"  total fractional mismatch        : {total_cost:.6g}")
    print(f"  max fractional mismatch          : {max_dist:.6g}")
    print(f"  ref indices in test order        : {perm_ref_in_test_order}")

    return perm_ref_in_test_order, origin_shift, max_dist, succes


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
    im = ax.imshow(O, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap='managua')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$| \langle u_\mathrm{ref} | u_\mathrm{test} \rangle |$", size=12)

    # ax.set_xlabel("MACE modes")    # Test modes
    # ax.set_ylabel("CRYSTAL modes") # Reference modes
    ax.set_xlabel(r"MACE modes in cm$^{-1}$", size=12)    # Test modes
    ax.set_ylabel(r"CRYSTAL modes in cm$^{-1}$", size=12) # Reference modes
    # ax.set_title(title)
    ax.text
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
    im = ax.imshow(O, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap='managua')
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

    xlabels, ylabels = group_label_modes(group_matches)
    
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90)

    ax.set_xlabel("Test groups")       # Test groups
    ax.set_ylabel("Reference groups")  # Reference groups
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_combined_overlap_heatmaps(
    overlap_matrix: np.ndarray,
    group_overlap_matrix: np.ndarray,
    group_matches: list,
    freqs_ref: np.ndarray,
    freqs_test: np.ndarray,
    skip_first: int = 3,
    outfile: str | Path = "combined_overlap_heatmaps.png",
):
    """
    Plot mode-overlap and degenerate-group overlap heatmaps side-by-side.

    Both subplots use:
        x-axis = CRYSTAL / reference
        y-axis = MACE / test
    """

    overlap_matrix = np.asarray(overlap_matrix, dtype=float)
    group_overlap_matrix = np.asarray(group_overlap_matrix, dtype=float)
    outfile = Path(outfile)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=False,
    )

    fig.subplots_adjust(wspace=0.005)
    # ============================================================
    # Left: single-mode overlap heatmap
    # Original overlap_matrix has:
    #   rows = CRYSTAL modes
    #   cols = MACE modes
    # Therefore transpose for:
    #   x = CRYSTAL
    #   y = MACE
    # ============================================================
    cmap = _slice_CMAP(CMAP)
    ax = axes[0]
    im1 = ax.imshow(
        overlap_matrix.T,
        origin="lower",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap
    )

    ax.set_xlabel(r"CRYSTAL modes in cm$^{-1}$")
    ax.set_ylabel(r"MACELES modes in cm$^{-1}$")
    # ax.set_xlabel(r"HSEsol modes in cm$^{-1}$")
    ax.xaxis.set_label_coords(0.5, -0.15)
    # ax.set_ylabel(r"PBEsol modes in cm$^{-1}$")



    ax.text(
        0.03,
        0.95,
        "a) Mode overlap",
        transform=ax.transAxes,
        va="top",
        ha="left",
        size=18,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # x-axis: CRYSTAL frequencies
    xlabels = [f"{f:.0f}" for f in freqs_ref[skip_first:]]
    step_x = max(1, len(xlabels) // 12)
    ax.set_xticks(np.arange(0, len(xlabels), step_x))
    ax.set_xticklabels(xlabels[::step_x], rotation=90)

    # y-axis: MACE frequencies
    ylabels = [f"{f:.0f}" for f in freqs_test[skip_first:]]
    step_y = max(1, len(ylabels) // 12)
    ax.set_yticks(np.arange(0, len(ylabels), step_y))
    ax.set_yticklabels(ylabels[::step_y])

    # ============================================================
    # Right: degenerate-group overlap heatmap
    # Original group_overlap_matrix has:
    #   rows = CRYSTAL groups
    #   cols = MACE groups
    # Therefore transpose for:
    #   x = CRYSTAL groups
    #   y = MACE groups
    # ============================================================

    ax = axes[1]

    im2 = ax.imshow(
        group_overlap_matrix.T,
        origin="lower",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap
    )

    cbar2 = fig.colorbar(im2, ax=ax)
    cbar2.set_label("Overlap")

    xlabels = []
    ylabels = []

    for g in group_matches:
        ref = g["ref_modes"]
        test = g["test_modes"]

        if len(ref) == 1:
            xlabels.append(f"{ref[0]}")
        else:
            xlabels.append(f"{ref[0]}-{ref[-1]}")

        if len(test) == 1:
            ylabels.append(f"{test[0]}")
        else:
            ylabels.append(f"{test[0]}-{test[-1]}")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90)

    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    ax.set_xlabel("CRYSTAL groups")
    ax.set_ylabel("MACELES groups")
    # ax.set_xlabel("HSEsol groups")
    ax.xaxis.set_label_coords(0.5, -0.15)
    # ax.set_ylabel("PBEsol groups")


    ax.text(
        0.03,
        0.95,
        "b) Degenerate-group overlap",
        transform=ax.transAxes,
        va="top",
        ha="left",
        size=18,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ============================================================
    # Save
    # ============================================================

    #fig.savefig(outfile, dpi=200, bbox_inches="tight")
    #fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
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


def compare_mode_sets(
    *,
    freqs_crys: np.ndarray,
    evecs_crys: np.ndarray,
    mace_modes: dict,
    skip_first: int = 3,
    degeneracy_tol: float = 0.5,
    heatmap_outfile: str | Path | None = None,
    title: str = "CRYSTAL vs MACE mode overlap",
    crystal_extra: dict | None = None,
    source: str = "unknown",
    mode = None
):
    freqs_crys = np.asarray(freqs_crys, dtype=float)
    evecs_crys = np.asarray(evecs_crys, dtype=float)

    freqs_mace = np.asarray(mace_modes["freqs_cm"], dtype=float)
    evecs_mace = np.asarray(mace_modes["eigvecs_mw"], dtype=float)

    if evecs_crys.shape != evecs_mace.shape:
        raise ValueError(
            f"Mode eigenvector shape mismatch: "
            f"CRYSTAL {evecs_crys.shape}, MACE {evecs_mace.shape}."
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

        comb_heatmap_outfile = heatmap_outfile.with_name(
            heatmap_outfile.stem + "_combined" + heatmap_outfile.suffix
        )
        # print(comb_heatmap_outfile)
        plot_combined_overlap_heatmaps(
            overlap_matrix=overlap_cut,
            group_overlap_matrix=group_overlap_matrix,
            group_matches=group_matches,
            freqs_ref=freqs_crys,
            freqs_test=freqs_mace,
            outfile=comb_heatmap_outfile,
        )

    elif heatmap_outfile is None:
        if mode == 'return_overlap_matrix':

            return overlap_cut

    return {
        "crystal": {} if crystal_extra is None else crystal_extra,
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
        "source": source,
    }


def run_mode_comparison(
    atoms: Atoms,
    mace_modes: dict,
    crystal_hess_path: str | Path,
    freq_out_path: str | Path,
    crystal_hessian_units: str = "hartree/bohr^2",
    skip_first: int = 3,
    degeneracy_tol: float = 0.5,
    heatmap_outfile: str | Path | None = None,
    title: str = "CRYSTAL vs MACE mode overlap",
):
    """
    Old Wokflow will be deleted at some point!!!
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

    perm_ref_in_test_order, origin_shift, max_mismatch = build_atom_permutation_by_fracpos(
        ref_atoms=crystal_atoms,
        test_atoms=atoms,
        tol=1e-2,
    )

    # H_crys is originally in CRYSTAL/ref atom order.
    # Reorder it into MACE/test atom order before diagonalization.
    H_crys = reorder_hessian(H_crys, perm_ref_in_test_order)

    Hmw_crys, eigvals_crys, evecs_crys, freqs_crys, imag_crys = diagonalize_hessian(
        H_crys,
        masses_amu,
        hessian_units=crystal_hessian_units,
    )

    return compare_mode_sets(
        freqs_crys=freqs_crys,
        evecs_crys=evecs_crys,
        mace_modes=mace_modes,
        skip_first=skip_first,
        degeneracy_tol=degeneracy_tol,
        heatmap_outfile=heatmap_outfile,
        title=title,
        crystal_extra={
            "H_cart": H_crys,
            "H_mw": Hmw_crys,
            "eigvals_SI": eigvals_crys,
            "eigvecs_mw": evecs_crys,
            "freqs_cm": freqs_crys,
            "imag_flags": imag_crys,
            "atom_permutation": perm_ref_in_test_order,
            "origin_shift": origin_shift,
            "max_atom_mismatch": max_mismatch,
        },
        source="hessfreq",
    )


def run_mode_comparison_from_ref_db(
    ref_db_path: str | Path,
    structure: str,
    mace_modes: dict,
    atoms: Atoms | None = None,
    skip_first: int = 3,
    degeneracy_tol: float = 0.5,
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
    print('Reading modes: succes')
    freqs_crys = np.asarray(crystal["freqs_cm"], dtype=float)
    evecs_crys = np.asarray(crystal["eigvecs_mw"], dtype=float)

    permutation_info = {}

    if atoms is not None:
        crystal_atoms = None

        if all(k in crystal for k in ("atomic_numbers", "cell_A", "scaled_positions")):
            crystal_atoms = Atoms(
                numbers=np.asarray(crystal["atomic_numbers"], dtype=int),
                cell=np.asarray(crystal["cell_A"], dtype=float),
                scaled_positions=np.asarray(
                    crystal["scaled_positions"],
                    dtype=float,
                ),
                pbc=True,
            )
            print('Building ref Atoms object: succes')
        else:
            print(
                "WARNING: ref_db mode comparison could not apply atom permutation "
                "because read_crystal_modes() did not return "
                "atomic_numbers/cell_A/scaled_positions."
            )

        if crystal_atoms is not None:
            for tol in [0.01, 0.015, 0.02, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060]:
                perm_ref_in_test_order, origin_shift, max_mismatch, succes = build_atom_permutation_by_fracpos(
                    ref_atoms=crystal_atoms,
                    test_atoms=atoms,
                    tol=tol,
                )
                if succes: 
                    print('Reordering atoms: succes')
                    break

            evecs_crys = reorder_mode_eigenvectors(evecs_crys, perm_ref_in_test_order)

            permutation_info = {
                "atom_permutation": perm_ref_in_test_order,
                "origin_shift": origin_shift,
                "max_atom_mismatch": max_mismatch,
            }
            
        else:
            print(
                "WARNING: ref_db mode comparison could not apply atom permutation "
                "because read_crystal_modes() did not return atoms/symbols/cell/scaled_positions."
            )

    if title is None:
        title = f"{structure}: CRYSTAL DB vs MACE mode overlap"

    crystal_extra = dict(crystal)
    crystal_extra.update(permutation_info)

    return compare_mode_sets(
        freqs_crys=freqs_crys,
        evecs_crys=evecs_crys,
        mace_modes=mace_modes,
        skip_first=skip_first,
        degeneracy_tol=degeneracy_tol,
        heatmap_outfile=heatmap_outfile,
        title=title,
        crystal_extra=crystal_extra,
        source="ref_db",
    )
