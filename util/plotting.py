import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from .ref_db import read_crystal_ir_reference

CMAP = mpl.colormaps['viridis']

def gaussian_profile(x, x0=0, intensity=1, fwhm=400):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return intensity * np.exp(-0.5*((x-x0)/sigma)**2)


def lorentzian_profile(x, x0=0, intensity=1, gamma=400):
    g_half_2 = (gamma/2)**2
    return intensity * g_half_2/((x-x0)**2+g_half_2)


def pseudo_voigt(gaussian, lorentzian, mixing=0.3):
    inverse = 1-mixing
    return mixing*gaussian+inverse*lorentzian


def perform_KDE(freqs, intnsities, FWHM=250):
    N = 2000                # number of points
    nu_space = np.linspace(freqs.min()-100, freqs.max()+100, N)
    total_spectrum_g = np.zeros_like(nu_space)
    total_spectrum_l = total_spectrum_g.copy()

    for i, n in zip(intnsities, freqs):
        total_spectrum_g += gaussian_profile(nu_space, n, i, fwhm=FWHM)
        total_spectrum_l += lorentzian_profile(nu_space, n, i, gamma=10)

    # voigt = pseudo_voigt(total_spectrum_g, total_spectrum_l, 0.5)

    return total_spectrum_g, nu_space


def restore_degeneracies(freqs, intensities, intensity_mode="sum"):
    """
    Reconstruct degeneracies and return only frequencies with intensity > 0.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (degenerate modes appear multiple times)
    intensities : np.ndarray
        Intensity array (same length)
    intensity_mode : str
        "sum"  -> sum intensities of degenerate modes
        "mean" -> average intensity of degenerate modes
        "max"  -> take maximum intensity of degenerate modes

    Returns
    -------
    out_freqs : np.ndarray
        Unique frequencies with intensity > 0
    out_intens : np.ndarray
        Intensities after combining degeneracies
    degeneracies : dict
        Dict mapping freq -> degeneracy
    """

    # Group modes by frequency (use rounding for safety)
    rounded = np.round(freqs, decimals=4)
    unique_freqs = np.unique(rounded)

    out_freqs = []
    out_intens = []
    degeneracies = {}

    for uf in unique_freqs:
        mask = (rounded == uf)
        deg = np.sum(mask)
        degeneracies[uf] = deg

        intens_group = intensities[mask]

        # skip if all intensities = 0
        if np.all(intens_group <= 0):
            continue

        # combine intensities depending on mode
        if intensity_mode == "sum":
            I = np.sum(intens_group)
        elif intensity_mode == "mean":
            I = np.mean(intens_group)
        elif intensity_mode == "max":
            I = np.max(intens_group)
        else:
            raise ValueError("intensity_mode must be 'sum', 'mean', or 'max'")

        out_freqs.append(uf)
        out_intens.append(I)

    return np.array(out_freqs), np.array(out_intens), degeneracies


def plot_ir_spectrum(
    freqs_cm,
    intensities,
    nu_grid,
    ir_spec,
    structure: str,
    crystal_db_path: str | Path = "/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5",
    outfile: str | Path = "ir_spectrum.pdf",
):
    cmap = CMAP
    c = cmap([0.4, 0.7])


    fig, (ax0,ax1) = plt.subplots(2, 1, sharex=True)
    try:
        f_crys, I_crys = read_crystal_ir_reference(crystal_db_path, structure)
        f_crys, I_crys, _ = restore_degeneracies(f_crys, I_crys)

        pos_mask = np.asarray(f_crys) > 1e-6
        f_crys = np.asarray(f_crys, dtype=float)[pos_mask]
        I_crys = np.asarray(I_crys, dtype=float)[pos_mask]

        if np.max(I_crys) > 0.0:
            I_crys_rel = I_crys / np.max(I_crys)
        else:
            I_crys_rel = I_crys.copy()

        # print('I_crys', I_crys)
        kde_ref, x_ref = perform_KDE(f_crys, I_crys_rel, 12.0)
        ax1.plot(nu_grid, ir_spec, lw=0.7, ls='--', label=f'MACELES {structure}', color=c[1])
        ax1.plot(x_ref, kde_ref, label='CRYSTAL', color=c[0])
        ax1.legend()
        
    except Exception as exc:
        print(f"----> No CRYSTAL reference spectrum found for {structure}: {exc}")
    # broadened spectrum

    ax0.plot(nu_grid, ir_spec, label=f'MACELES {structure}', color=c[1])
    ax0.legend()

    ax1.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax1.set_ylabel("Relative IR intensity")
    ax0.set_ylabel("Relative IR intensity")
    
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close()

    return f_crys, I_crys_rel, x_ref, kde_ref


def plot_ir_spectrum_with_frequency_correlation(
    freqs_cm,
    intensities,
    nu_grid,
    ir_spec,
    structure: str,
    crystal_freqs_cm=None,
    crystal_db_path: str | Path = "/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5",
    outfile: str | Path = "ir_spectrum_frequency_correlation.pdf",
):
    """
    Like plot_ir_spectrum(), but adds a right-side phonon-frequency
    correlation axis spanning both IR spectrum axes.

    Left:
        top    -> MACELES broadened IR spectrum
        bottom -> MACELES vs CRYSTAL broadened IR spectrum

    Right:
        scatter plot of CRYSTAL frequencies vs MACELES frequencies
        with dashed y = x reference line.

    Notes
    -----
    This does NOT do mode matching.
    It simply compares frequencies by sorted order after filtering
    positive modes.
    """

    cmap = CMAP
    c = cmap([0.4, 0.7, 0.8])

    freqs_cm = np.asarray(freqs_cm, dtype=float)

    f_crys = None
    I_crys_rel = None
    x_ref = None
    kde_ref = None

    fig = plt.figure(figsize=(15, 5))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[2.5, 1.3],
        height_ratios=[1, 1],
        wspace=0.05,
        hspace=0.08,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax_corr = fig.add_subplot(gs[:, 1])

    # ------------------------------------------------------------
    # Read CRYSTAL reference
    # ------------------------------------------------------------
    try:
        f_crys, I_crys = read_crystal_ir_reference(crystal_db_path, structure)
        f_crys, I_crys, _ = restore_degeneracies(f_crys, I_crys)

        pos_mask = np.asarray(f_crys) > 1e-6
        f_crys = np.asarray(f_crys, dtype=float)[pos_mask]
        I_crys = np.asarray(I_crys, dtype=float)[pos_mask]

        if len(I_crys) and np.max(I_crys) > 0.0:
            I_crys_rel = I_crys / np.max(I_crys)
        else:
            I_crys_rel = I_crys.copy()

        kde_ref, x_ref = perform_KDE(f_crys, I_crys_rel, 12.0)

    except Exception as exc:
        print(f"----> No CRYSTAL reference spectrum found for {structure}: {exc}")

    # Optional override if you already have full CRYSTAL phonon frequencies
    if crystal_freqs_cm is not None:
        f_corr_crys = np.asarray(crystal_freqs_cm, dtype=float)
    elif f_crys is not None:
        f_corr_crys = np.asarray(f_crys, dtype=float)
    else:
        f_corr_crys = None

    # ------------------------------------------------------------
    # Top spectrum: MACELES only
    # ------------------------------------------------------------
    ax0.plot(
        nu_grid,
        ir_spec,
        label=f"MACELES {structure}",
        color=c[1],
    )
    ax0.legend()
    plt.setp(ax0.get_xticklabels(), visible=False)

    # ------------------------------------------------------------
    # Bottom spectrum: MACELES vs CRYSTAL
    # ------------------------------------------------------------
    ax1.plot(
        nu_grid,
        ir_spec,
        lw=0.7,
        ls="--",
        label=f"MACELES {structure}",
        color=c[1],
    )

    if x_ref is not None and kde_ref is not None:
        ax1.plot(
            x_ref,
            kde_ref,
            label="CRYSTAL",
            color=c[0],
        )

    ax1.legend()
    ax1.set_xlabel(r"Wavenumber in cm$^{-1}$")
    
    fig.supylabel("Relative IR intensity", x=0.078, size=14)
    # ------------------------------------------------------------
    # Right axis: frequency-correlation scatter
    # ------------------------------------------------------------
    ml_freqs_pos = np.asarray(freqs_cm, dtype=float)
    ml_freqs_pos = ml_freqs_pos[ml_freqs_pos > 1e-6]
    ml_freqs_pos = np.sort(ml_freqs_pos)

    if f_corr_crys is not None:
        crys_freqs_pos = np.asarray(f_corr_crys, dtype=float)
        crys_freqs_pos = crys_freqs_pos[crys_freqs_pos > 1e-6]
        crys_freqs_pos = np.sort(crys_freqs_pos)

        n_compare = min(len(crys_freqs_pos), len(ml_freqs_pos))

        if n_compare > 0:
            x = crys_freqs_pos[:n_compare]
            y = ml_freqs_pos[:n_compare]

            ax_corr.scatter(
                x,
                y,
                s=83,
                marker='d',
                alpha=0.66,
                edgecolors=cmap([0.3]),
                linewidths = 0.75,
                color = c[2],
                label=r"$\Gamma$-Mode correlation",
            )

            f_min = min(np.min(x), np.min(y))
            f_max = max(np.max(x), np.max(y))

            pad = 0.05 * (f_max - f_min) if f_max > f_min else 10.0
            f_min -= pad
            f_max += pad

            ax_corr.plot(
                [f_min, f_max],
                [f_min, f_max],
                ls="--",
                lw=0.7,
                color="grey",
            )

            ax_corr.set_xlim(f_min, f_max)
            ax_corr.set_ylim(f_min, f_max)
            ax_corr.set_aspect("equal", adjustable="box")

        else:
            ax_corr.text(
                0.5,
                0.5,
                "No positive\nfrequencies",
                ha="center",
                va="center",
                transform=ax_corr.transAxes,
            )

    else:
        ax_corr.text(
            0.5,
            0.5,
            "No CRYSTAL\nfrequencies",
            ha="center",
            va="center",
            transform=ax_corr.transAxes,
        )

    ax_corr.legend()
    ax_corr.set_xlabel("CRYSTAL frequency in cm$^{-1}$")
    ax_corr.set_ylabel("MACELES frequency in cm$^{-1}$")
    ticks = ax_corr.get_xticks()
    print(ticks)
    ax_corr.set_yticks(ticks[1:-1])
    ax_corr.yaxis.set_label_position("right")
    ax_corr.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)
    # ax_corr.grid()

    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "f_crys": f_crys,
        "I_crys_rel": I_crys_rel,
        "x_ref": x_ref,
        "kde_ref": kde_ref,
    }
