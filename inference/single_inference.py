#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import eigh
from ase.io import read
from ase.optimize import LBFGS
from ase.units import _amu
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
from pathlib import Path
import sys
PYTHON_SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_SCRIPTS_ROOT))
from TOOLs.PlotVibrationsModule import plot_ir_spectrum
from TOOLs.IR_CORE import *
from TOOLs.ModeComparison import *
import matplotlib as mpl
mpl.rcParams['font.size'] = 14

CMAP = mpl.colormaps['viridis']

ROOT = '/home/jha/jha/python_scripts/CRYSTALdataGen/'

FRECHET = True

###############   TiO2_anatase     ###################
# STRUCT = 'TiO2_I41amd'
# MODEL_PATH = ROOT + 'models/MACELES_.model'


###############   TiO2_rutil       ###################
# STRUCT = 'TiO2_rutil'
# MODEL_PATH = ROOT + 'models/TiO2_650.model'
# HESS_PATH = Path('/home/jha/jha/python_scripts/CRYSTALreference/00_structures/TiO2_rutil/freq/TiO2_rutil_freq.hessfreq')
# FREQ_OUT_PATH = Path('/home/jha/jha/python_scripts/CRYSTALreference/00_structures/TiO2_rutil/freq/TiO2_rutil_freq.out')

######################################################
# STRUCT = "SiO2"
# MODEL_PATH = ROOT + 'models/SiO2_bestof18.model' # 650
# MODEL_PATH = ROOT + 'models/SiO2_comb_450.model'        # 450
# MODEL_PATH = ROOT + 'models/SiO2comb.model'        # 250
# HESS_PATH = Path('/home/jha/jha/python_scripts/CRYSTALreference/00_structures/SiO2/freq/SiO2_freq.hessfreq')
# FREQ_OUT_PATH = Path('/home/jha/jha/python_scripts/CRYSTALreference/00_structures/SiO2/freq/SiO2_freq.out')
# SiO2 needs more data


######################################################
STRUCT = "Al2O3"
MODEL_PATH = ROOT + 'models/Al2O3_comb.model'
HESS_PATH = Path('/home/jha/jha/python_scripts/CRYSTALreference/00_structures/Al2O3/freq/Al2O3_freq.hessfreq')
FREQ_OUT_PATH = Path('/home/jha/jha/python_scripts/CRYSTALreference/00_structures/Al2O3/freq/Al2O3_freq.out')


######################################################
# STRUCT = "AlN"
# MODEL_PATH = ROOT + 'models/AlN_MACELES.model'
# HESS_PATH = Path('/home/jha/jha/python_scripts/CRYSTALreference/00_structures/AlN/freq/AlN_freq.hessfreq')
# FREQ_OUT_PATH = Path('/home/jha/jha/python_scripts/CRYSTALreference/00_structures/AlN/freq/AlN_freq.out')

CIFROOT = ROOT + 'inference/CIFs/'
PLOTROOT = ROOT + 'inference/Plots/'
MODE_OVERLAP_PLOT = PLOTROOT + STRUCT + "_mode_overlap.png"

MAIN = True
COMPARE_TO_CRYSTAL = True
CRYSTAL_HESSIAN_UNITS = "hartree/bohr^2"


def calculator(cal='0'):
    if cal == '0':
        from mace.calculators import MACECalculator
        calc = MACECalculator(
            model_paths=MODEL_PATH,
            default_dtype="float64",
            device="cuda",
        )
        return calc
    else:
        print('Calculator not available!!')


def geometry_optimisation(atoms, fmax=1e-3):
    ei = atoms.get_potential_energy()
    print("Initial Energy:", ei, "eV")

    if FRECHET:
        atoms.set_constraint(FixSymmetry(atoms))
        ecf = FrechetCellFilter(atoms)
        opt = LBFGS(ecf, trajectory="Opt.traj")
    else: 
        opt = LBFGS(atoms, trajectory="Opt.traj")

    opt.run(fmax=fmax)

    ef = atoms.get_potential_energy()
    print("Final Energy:", ef, "eV")
    return atoms


def get_les_outputs(calc, atoms, compute_forces=False, compute_stress=False):
    """
    Directly call the underlying MACELES model and request LES outputs.
    """
    batch = calc._atoms_to_batch(atoms)
    model = calc.models[0]

    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    for key in batch.keys:
        val = batch[key]
        if torch.is_tensor(val):
            val = val.to(model_device)
            if torch.is_floating_point(val):
                val = val.to(dtype=model_dtype)
            batch[key] = val

    model.eval()
    out = model(
        batch.to_dict(),
        training=False,
        compute_force=compute_forces,
        compute_stress=compute_stress,
        compute_bec=True,
    )

    out_np = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            out_np[k] = v.detach().cpu().numpy()
        else:
            out_np[k] = v
    return out_np


def main(file, struct):
    atoms = read(file, format="cif")
    atoms.set_pbc(True)
    atoms = standardize_to_primitive(atoms, no_idealize=False)

    if MAIN:
        calc = calculator()
        atoms.calc = calc
        print('\n\n##                   Model eval                   ##')
        # Geometry optimization
        atoms = geometry_optimisation(atoms, fmax=1e-11)

        # LES / BEC extraction
        print("\n=== LES outputs ===")
        les_out = get_les_outputs(calc, atoms, compute_forces=False, compute_stress=False)

        # print("Available keys:", sorted(les_out.keys()))
        if "les_energy" in les_out:
            print("LES energy:", les_out["les_energy"])
        if "latent_charges" in les_out and les_out["latent_charges"] is not None:
            print("latent_charges shape:", np.asarray(les_out["latent_charges"]).shape)
            print("latent_charges:")
            print(les_out["latent_charges"])

        bec = les_out.get("BEC", None)
        # print_bec_summary(bec)
        bec = np.asarray(bec, dtype=float)

        # Frequencies from analytical Hessian
        print("\n=== Hessian / frequencies ===")
        mace_modes = get_full_hessian_modes_from_calc(
            atoms,
            calc,
            hessian_units="eV/Ang^2",
            tol_eig_negative=1e-8,
        )

        freqs_cm = mace_modes["freqs_cm"]
        eigvecs = mace_modes["eigvecs_mw"]
        imag_flags = mace_modes["imag_flags"]
        zpe_eV = mace_modes["zpe_eV"]
        eigvals_SI = mace_modes["eigvals_SI"]

        for i, f in enumerate(freqs_cm):
            tag = "i" if imag_flags[i] else " "
            print(f"mode #{i:3d}: {f:10.4f}{tag} cm^-1")
        print("ZPE (eV):", zpe_eV)

        masses = atoms.get_masses()

        bec_asr = asr_correct_bec(bec)

        eigvecs_cart = mass_unweight_eigenvectors(eigvecs, masses)
        eigvecs_cart = normalize_modes_cartesian(eigvecs_cart)

        z_mode, intensities = mode_effective_charges(bec_asr, eigvecs_cart)

        print_ir_modes(freqs_cm, imag_flags, intensities, z_mode)

        nu_grid, ir_spec = broaden_spectrum(freqs_cm, intensities, imag_flags=imag_flags, fwhm=12.0)

        outfile = PLOTROOT + struct + '_new' + '.png'
        plot_ir_spectrum(freqs_cm, intensities, nu_grid, ir_spec, struct, outfile=outfile)


    # ============================================================
    # CRYSTAL mode comparison
    # ============================================================
    if COMPARE_TO_CRYSTAL and HESS_PATH is not None:
        try:
            if Path(HESS_PATH).exists():

                comparison = run_mode_comparison(
                    atoms=atoms,
                    mace_modes=mace_modes,
                    crystal_hess_path=HESS_PATH,
                    freq_out_path=FREQ_OUT_PATH,
                    crystal_hessian_units=CRYSTAL_HESSIAN_UNITS,
                    skip_first=3,
                    degeneracy_tol=1.0,
                    heatmap_outfile=MODE_OVERLAP_PLOT,
                    title=f"{struct}: CRYSTAL vs MACE mode overlap",
                )

                print_mode_match_summary(comparison["matches"])
                print_group_match_summary(comparison["subgroups"])

                if comparison.get("group_heatmap_outfile") is not None:
                    print(f"Saved degenerate-group heatmap to: {comparison['group_heatmap_outfile']}")

                print("\nDegenerate/subspace comparison:")
                print(f"{'g_ref':>6s} {'g_test':>6s} {'ref_modes':>18s} {'test_modes':>18s} {'subspace_overlap':>18s}")
                print("-" * 70)

                for g in comparison["subgroups"]:
                    ref_modes = [int(x + 1) for x in g["ref_modes"]]
                    test_modes = [int(x + 1) for x in g["test_modes"]]
                    print(
                        f"{g['group_ref_index']+1:6d} "
                        f"{g['group_test_index']+1:6d} "
                        f"{str(ref_modes):>18s} "
                        f"{str(test_modes):>18s} "
                        f"{g['subspace_overlap']:18.6f}"
                    )

                print(f"\nSaved mode-overlap heatmap to: {MODE_OVERLAP_PLOT}")

            else:
                print(f"\nNo CRYSTAL Hessian found at: {HESS_PATH}")
        except Exception as exc:
            print("\nMode comparison against CRYSTAL failed:")
            print(exc)


    #atoms.calc = None
    #del calc
    #torch.cuda.empty_cache()
    #torch.cuda.ipc_collect()


if __name__ == "__main__":
    struct = STRUCT
    file_name = CIFROOT + struct + '.cif'
    main(file_name, struct)
