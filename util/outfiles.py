import os
import re
import numpy as np
from pathlib import Path
from .triple_zeta import triple_zeta_basis as TZVP

# ============================================================
# QSUBS
# ============================================================
def write_sp_qsub_RUNE(folder: Path, structure_name: str, ncores, rune=None) -> None:
    # #SBATCH --nodelist=rune0{RUNE}
    # CRY23_SCRDIR="/scratch_rune0{RUNE}/jha/"
    rune_nr = f'\n#SBATCH --nodelist=rune0{rune}' if rune is not None else ''

    qsub = f'''#!/bin/bash

#SBATCH --partition=CPU_rune{rune_nr}
#SBATCH -J {structure_name}-SP
#SBATCH -N 1
#SBATCH --ntasks-per-node={ncores}
#SBATCH --cpus-per-task=1

source /usr/local/_tci_software_environment_.sh
module purge
module load 'rune/crystal/23_v1.0.1_omp'

export CRY23_SCRDIR="/scratch_rune0{rune if rune is not None else '5'}/jha/"


runPcry23OMP {ncores} {structure_name}_sp
'''
    out = folder / 'qsub.sh'
    out.write_text(qsub)
    os.chmod(out, 0o755)


def write_sp_qsub_AQ(folder: Path, structure_name: str, ncores) -> None:
    # #SBATCH --nodelist=rune0{RUNE}
    # CRY23_SCRDIR="/scratch_rune0{RUNE}/jha/"

    qsub = f'''#!/bin/bash

#SBATCH --partition=GPU_aq
#SBATCH -J {structure_name}-SP
#SBATCH -N 1
#SBATCH --ntasks-per-node={ncores}
#SBATCH --cpus-per-task=1

source /usr/local/_tci_software_environment_.sh
module purge
module load 'rune/crystal/23_v1.0.1_omp'

mkdir /scratch/jha
export CRY23_SCRDIR=/scratch/jha

runPcry23OMP {ncores} {structure_name}_sp
'''
    out = folder / 'qsub.sh'
    out.write_text(qsub)
    os.chmod(out, 0o755)


# ============================================================
# CRYSTAL writing
# ============================================================
def base_structure_name(structure_name: str) -> str:
    for suffix in ("_PBE", "_PBESOLXC", "_PBESOL", "_HSESOL", "_HSE"):
        if structure_name.upper().endswith(suffix):
            return structure_name[:-len(suffix)]
    return structure_name


def get_formula_part(name: str) -> str:
    return name.split("_", 1)[0]


def get_formula_elements(formula_name: str) -> list[str]:
    formula = get_formula_part(formula_name)
    return re.findall(r"[A-Z][a-z]?(?=\d|[A-Z]|$)", formula)


def get_basesets(formula_name: str) -> str:
    elements = get_formula_elements(formula_name)
    basesets = ''
    for elem in elements:
        if elem not in TZVP:
            raise KeyError(f'No TZVP basis found for element: {elem}')
        basesets += TZVP[elem]
    return basesets


def get_dft_block(*params) -> str:
    FUNCTIONAL, SHRINK, MAXCYCLE, USE_GRADCAL, MULLIKEN, SAVEWF = params
    lines = [
        '\n99 0',
        'END',
        'DFT',
        f'{FUNCTIONAL}',
        'END',
        'SHRINK',
        f'{SHRINK[0]} {SHRINK[1]}',
        'TOLINTEG',
        '7 7 7 9 30',
        'MAXCYCLE',
        str(MAXCYCLE),
    ]
    if USE_GRADCAL:
        lines.append('GRADCAL')
    if MULLIKEN:
        lines.append('PPAN')
    if SAVEWF:
        lines.append('SAVEWF')
    lines.append('END')
    return '\n'.join(lines) + '\n'


def cell_to_crystal_p1_parameters(cell: np.ndarray) -> str:
    a_vec, b_vec, c_vec = np.asarray(cell)

    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    alpha = np.degrees(np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1.0, 1.0)))
    beta  = np.degrees(np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1.0, 1.0)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1.0, 1.0)))

    return f'{a:.10f} {b:.10f} {c:.10f} {alpha:.10f} {beta:.10f} {gamma:.10f}'


def generate_crystal_block_from_atoms(atoms: Atoms, system_name: str) -> str:
    """
    Write explicit P1 geometry.
    """
    if not np.all(atoms.pbc):
        raise ValueError('Atoms object must be periodic in all directions.')

    scaled = atoms.get_scaled_positions(wrap=True)
    numbers = atoms.get_atomic_numbers()
    natoms = len(atoms)

    lines = []
    lines.append(f'{system_name}  P1')
    lines.append('CRYSTAL')
    lines.append('0 0 0')
    lines.append('1')
    lines.append(cell_to_crystal_p1_parameters(atoms.cell.array))
    lines.append(str(natoms))

    for Z, (x, y, z) in zip(numbers, scaled):
        Z_crystal = Z + 200 if 35 < Z < 100 else Z
        lines.append(f'{Z_crystal:3d} {x:.10f} {y:.10f} {z:.10f}')

    return '\n'.join(lines)


def write_singlepoint_d12(folder: Path, structure_name: str, geo_txt: str, basesets: str, dft_block: str) -> None:
    text = geo_txt + '\nEND' + basesets + dft_block
    (folder / f'{structure_name}_sp.d12').write_text(text)


def write_submit_all(folder: Path) -> None:
    script = '''#!/bin/bash
set -e

count=0
while IFS= read -r -d '' qsub; do
    dir=$(dirname "$qsub")
    echo "Submitting in $dir"
    (
        cd "$dir"
        sbatch qsub.sh
    )
    count=$((count + 1))
done < <(find . -type f -name "qsub.sh" -print0 | sort -z)

echo "Submitted $count jobs."
'''
    path = folder / 'submit_all.sh'
    path.write_text(script)
    os.chmod(path, 0o755)

