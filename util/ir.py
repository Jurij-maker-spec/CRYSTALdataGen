import numpy as np


def asr_correct_bec(bec):
    bec = np.asarray(bec, dtype=float)
    return bec - bec.sum(axis=0, keepdims=True) / bec.shape[0]


def mass_unweight_eigenvectors(eigvecs_mw, masses_amu):
    masses_amu = np.asarray(masses_amu, dtype=float)
    m_sqrt_inv = 1.0 / np.sqrt(np.repeat(masses_amu, 3))
    return m_sqrt_inv[:, None] * eigvecs_mw


def normalize_modes_cartesian(eigvecs_cart):
    out = eigvecs_cart.copy()
    for i in range(out.shape[1]):
        nrm = np.linalg.norm(out[:, i])
        if nrm > 0:
            out[:, i] /= nrm
    return out


def mode_effective_charges(bec, eigvecs_cart):
    n_atoms = bec.shape[0]
    n_modes = eigvecs_cart.shape[1]

    z_mode = np.zeros((n_modes, 3), dtype=float)
    intensities = np.zeros(n_modes, dtype=float)

    for m in range(n_modes):
        vec = eigvecs_cart[:, m].reshape(n_atoms, 3)
        zm = np.zeros(3, dtype=float)

        for a in range(n_atoms):
            zm += bec[a] @ vec[a]

        z_mode[m] = zm
        intensities[m] = np.dot(zm, zm)

    return z_mode, intensities


def broaden_spectrum(
    freqs_cm,
    intensities,
    imag_flags=None,
    fwhm=12.0,
    npts=4000,
    pad=100.0,
    unit="normal",
    threshold=1e-8,
    print_modes=False
):
    freqs_cm = np.asarray(freqs_cm, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if imag_flags is None:
        imag_flags = np.zeros_like(freqs_cm, dtype=bool)
    else:
        imag_flags = np.asarray(imag_flags, dtype=bool)

    if not (len(freqs_cm) == len(intensities) == len(imag_flags)):
        raise ValueError("freqs_cm, intensities, and imag_flags must have the same length.")

    phys_mask = (~imag_flags) & (freqs_cm > 1e-6)

    if not np.any(phys_mask):
        nu_grid = np.linspace(0.0, max(pad, 1.0), npts)
        spec = np.zeros_like(nu_grid)
        return nu_grid, spec

    nu_phys = freqs_cm[phys_mask]
    inten_phys = intensities[phys_mask]

    ir_mask = inten_phys > threshold
    nu = nu_phys[ir_mask]
    inten = inten_phys[ir_mask]

    nu_min = max(0.0, nu_phys.min() - pad)
    nu_max = nu_phys.max() + pad
    nu_grid = np.linspace(nu_min, nu_max, npts)

    if len(nu) == 0:
        spec = np.zeros_like(nu_grid)
        return nu_grid, spec

    if print_modes:
        print("\nModes used for broadened IR spectrum:")
        print(f"{'freq/cm^-1':>12s} {'IR_int':>14s}")
        for f, I in zip(nu, inten):
            print(f"{f:12.4f} {I:14.6e}")

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spec = np.zeros_like(nu_grid)

    for nui, Ii in zip(nu, inten):
        spec += Ii * np.exp(-0.5 * ((nu_grid - nui) / sigma) ** 2)

    if unit == "normal":
        smax = spec.max()
        if smax > 0.0:
            spec = spec / smax

    return nu_grid, spec


def print_ir_modes(freqs_cm, imag_flags, intensities, z_mode, threshold=1e-8):
    print("\n=== Harmonic IR mode analysis ===")
    print(f"{'mode':>4s} {'freq/cm^-1':>12s} {'IR_int':>14s} {'|Z_mode|':>14s}")

    for i, (f, imag) in enumerate(zip(freqs_cm, imag_flags)):
        if imag or f <= 1e-6:
            continue
        zmag = np.linalg.norm(z_mode[i])
        I = intensities[i]
        active = "*" if I > threshold else " "
        print(f"{i:4d} {f:12.4f} {I:14.6e} {zmag:14.6e} {active}")


def print_bec_summary(bec):
    if bec is None:
        print("BEC output is None")
        return

    bec = np.asarray(bec)
    print("BEC shape:", bec.shape)

    if bec.ndim != 3 or bec.shape[1:] != (3, 3):
        print("Warning: unexpected BEC shape")
        return

    for i in range(len(bec)):
        print(f"\nAtom {i} BEC:")
        print(bec[i])

    print("\nAcoustic sum rule check Σ_i Z*_i:")
    print(bec.sum(axis=0))

    bec_corr = bec - bec.sum(axis=0, keepdims=True) / bec.shape[0]

    print("ASR-corrected sum:\n", bec_corr.sum(axis=0))
    for i in range(len(bec_corr)):
        print(f"\nAtom {i} corrected BEC:\n{bec_corr[i]}")

    print("atom0-atom1:\n", bec_corr[0] - bec_corr[1])
    print("atom2-atom3:\n", bec_corr[2] - bec_corr[3])


