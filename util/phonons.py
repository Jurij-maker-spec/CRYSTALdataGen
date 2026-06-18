import numpy as np
import spglib
from scipy import constants as sc
from scipy.linalg import eigh
from ase import Atoms
from ase.io import read
from ase.units import _amu

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy import load


def ase_atoms_to_phonopy_atoms(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )


def phonopy_atoms_to_ase_atoms(ph_atoms):
    return Atoms(
        symbols=ph_atoms.symbols,
        cell=np.array(ph_atoms.cell),
        scaled_positions=np.array(ph_atoms.scaled_positions),
        pbc=True,
    )


def get_primitive_atoms_from_cif(file):
    atoms = read(file, format="cif")
    atoms.set_pbc(True)

    cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    primitive = spglib.standardize_cell(cell, to_primitive=True, no_idealize=False)

    if primitive is None:
        raise RuntimeError("spglib.standardize_cell returned None")

    prim_atoms = Atoms(
        numbers=primitive[2],
        scaled_positions=primitive[1],
        cell=primitive[0],
        pbc=True
    )
    return prim_atoms


def freqs_from_analytical_hessian(
        atoms, 
        calc, 
        hessian_units="eV/Ang^2", 
        tol_eig_negative=1e-8, 
        print_freqs=False
    ):
    
    H_raw = calc.get_hessian(atoms=atoms)
    H = np.asarray(H_raw).reshape(3 * len(atoms), 3 * len(atoms))

    if hessian_units != "eV/Ang^2":
        raise ValueError("Only 'eV/Ang^2' is supported")

    eV_to_J = sc.e
    Ang_to_m = 1e-10
    H_SI = H * eV_to_J / (Ang_to_m**2)

    masses_amu = atoms.get_masses()
    masses_kg = masses_amu * _amu

    m = np.repeat(masses_kg, 3)
    H_mw = H_SI / np.sqrt(np.outer(m, m))
    H_mw = 0.5 * (H_mw + H_mw.T)

    eigvals_SI, eigvecs = eigh(H_mw)

    conv_to_wavenumber = 1.0 / (2.0 * np.pi * sc.c * 100.0)

    freqs_cm = np.zeros_like(eigvals_SI)
    imag_flags = np.zeros_like(eigvals_SI, dtype=bool)

    for i, val in enumerate(eigvals_SI):
        if val >= 0.0:
            freqs_cm[i] = np.sqrt(val) * conv_to_wavenumber
        else:
            if abs(val) <= tol_eig_negative:
                freqs_cm[i] = 0.0
            else:
                freqs_cm[i] = np.sqrt(abs(val)) * conv_to_wavenumber
                imag_flags[i] = True

    positive_mask = (~imag_flags) & (freqs_cm > 0.0)
    if positive_mask.any():
        zpe_J = 0.5 * np.sum(sc.h * sc.c * (freqs_cm[positive_mask] * 100.0))
        zpe_eV = zpe_J / sc.e
    else:
        zpe_eV = 0.0

    if print_freqs:
        for i, f in enumerate(freqs_cm):
            tag = "i" if imag_flags[i] else " "
            print(f"mode #{i:3d}: {f:10.4f}{tag} cm^-1")
        print("ZPE (eV):", zpe_eV)

    return freqs_cm, eigvecs, imag_flags, zpe_eV, eigvals_SI


def hessian_unit_to_SI_factor(hessian_units: str) -> float:
    """
    Return conversion factor from Hessian units to J/m^2.

    Supported:
    - eV/Ang^2
    - hartree/bohr^2
    """
    hu = hessian_units.strip().lower()

    if hu == "ev/ang^2":
        return sc.e / (1.0e-10 ** 2)

    if hu in {"hartree/bohr^2", "ha/bohr^2"}:
        hartree = sc.physical_constants["Hartree energy"][0]
        bohr = sc.physical_constants["Bohr radius"][0]
        return hartree / (bohr ** 2)

    raise ValueError(f"Unsupported Hessian units: {hessian_units}")


def mass_weight_hessian(H_cart: np.ndarray, masses_amu: np.ndarray, hessian_units: str) -> np.ndarray:
    """
    Convert Cartesian Hessian to SI and mass-weight it.

    Returns
    -------
    H_mw : (3N, 3N) ndarray
        Mass-weighted Hessian in s^-2.
    """
    H_cart = np.asarray(H_cart, dtype=float)
    masses_amu = np.asarray(masses_amu, dtype=float)

    factor = hessian_unit_to_SI_factor(hessian_units)
    H_SI = H_cart * factor

    masses_kg = masses_amu * sc.atomic_mass
    m = np.repeat(masses_kg, 3)

    H_mw = H_SI / np.sqrt(np.outer(m, m))
    H_mw = 0.5 * (H_mw + H_mw.T)
    return H_mw


def eigvals_to_freqs_cm1(eigvals_SI: np.ndarray, tol_eig_negative: float = 1e-8):
    """
    Convert eigenvalues of a mass-weighted Hessian (s^-2) to cm^-1.

    Returns
    -------
    freqs_cm : ndarray
    imag_flags : ndarray(bool)
    """
    eigvals_SI = np.asarray(eigvals_SI, dtype=float)
    conv = 1.0 / (2.0 * np.pi * sc.c * 100.0)

    freqs_cm = np.zeros_like(eigvals_SI)
    imag_flags = np.zeros_like(eigvals_SI, dtype=bool)

    for i, val in enumerate(eigvals_SI):
        if val >= 0.0:
            freqs_cm[i] = np.sqrt(val) * conv
        else:
            if abs(val) <= tol_eig_negative:
                freqs_cm[i] = 0.0
            else:
                freqs_cm[i] = np.sqrt(abs(val)) * conv
                imag_flags[i] = True

    return freqs_cm, imag_flags


def diagonalize_hessian(
    H_cart: np.ndarray,
    masses_amu: np.ndarray,
    hessian_units: str,
    tol_eig_negative: float = 1e-8,
):
    """
    Diagonalize a Cartesian Hessian with the same convention as your
    freqs_from_analytical_hessian function.

    Returns
    -------
    H_mw : ndarray
    eigvals_SI : ndarray
    eigvecs_mw : ndarray
    freqs_cm : ndarray
    imag_flags : ndarray(bool)
    """
    H_mw = mass_weight_hessian(H_cart, masses_amu, hessian_units=hessian_units)
    eigvals_SI, eigvecs_mw = eigh(H_mw)
    freqs_cm, imag_flags = eigvals_to_freqs_cm1(
        eigvals_SI, tol_eig_negative=tol_eig_negative
    )
    return H_mw, eigvals_SI, eigvecs_mw, freqs_cm, imag_flags


def get_full_hessian_modes_from_calc(
    atoms: Atoms,
    calc,
    hessian_units: str = "eV/Ang^2",
    tol_eig_negative: float = 1e-8,
):
    """
    Companion to freqs_from_analytical_hessian that also returns H_cart and H_mw.
    """
    H_raw = calc.get_hessian(atoms=atoms)
    H_cart = np.asarray(H_raw, dtype=float).reshape(3 * len(atoms), 3 * len(atoms))
    H_cart = 0.5 * (H_cart + H_cart.T)

    masses_amu = atoms.get_masses()
    H_mw, eigvals_SI, eigvecs_mw, freqs_cm, imag_flags = diagonalize_hessian(
        H_cart,
        masses_amu,
        hessian_units=hessian_units,
        tol_eig_negative=tol_eig_negative,
    )

    positive_mask = (~imag_flags) & (freqs_cm > 0.0)
    if positive_mask.any():
        zpe_J = 0.5 * np.sum(sc.h * sc.c * (freqs_cm[positive_mask] * 100.0))
        zpe_eV = zpe_J / sc.e
    else:
        zpe_eV = 0.0

    return {
        "H_cart": H_cart,
        "H_mw": H_mw,
        "eigvals_SI": eigvals_SI,
        "eigvecs_mw": eigvecs_mw,
        "freqs_cm": freqs_cm,
        "imag_flags": imag_flags,
        "zpe_eV": zpe_eV,
    }


