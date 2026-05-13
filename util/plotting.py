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
