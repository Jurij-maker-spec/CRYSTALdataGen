from __future__ import annotations

import shutil
from pathlib import Path
from ase import Atoms
from ase.io import write

from .outfiles import (
    base_structure_name,
    get_basesets,
    get_dft_block,
    generate_crystal_block_from_atoms,
    write_singlepoint_d12,
    write_sp_qsub_RUNE,
    write_sp_qsub_AQ,
    write_submit_all,
)


def default_out_tag(cfg: dict) -> str:
    if cfg.get("OUT_TAG"):
        return cfg["OUT_TAG"]

    mode = cfg["DISTORTION_MODE"]
    batch = cfg["BATCH"]
    n = cfg["N_STRUCTURES"]

    return f"sp_{mode}_{batch}_{n}"


def build_output_root(structure_dir: Path, cfg: dict) -> Path:
    out_name = default_out_tag(cfg)

    if cfg["SUPERCELL"] is not None:
        sc = cfg["SUPERCELL"]
        out_name += f"_sc{sc[0]}{sc[1]}{sc[2]}"

    return structure_dir / "disto" / out_name


def make_output_folder(base_dir: Path, distortion_id: str, overwrite: bool) -> Path:
    folder = base_dir / distortion_id

    if folder.exists():
        if overwrite:
            shutil.rmtree(folder)
        else:
            raise FileExistsError(f"{folder} already exists. Set overwrite: true to replace.")

    folder.mkdir(parents=True, exist_ok=False)
    return folder


def prepare_crystal_blocks(structure_name: str, cfg: dict) -> tuple[str, str]:
    chem_comp = base_structure_name(structure_name)

    basesets = get_basesets(chem_comp)

    dft_block = get_dft_block(
        cfg["FUNCTIONAL"],
        cfg["SHRINK"],
        cfg["MAXCYCLE"],
        cfg["USE_GRADCAL"],
        cfg["MULLIKEN"],
        cfg["SAVEWF"],
    )

    return chem_comp, basesets, dft_block


def write_singlepoint_package(
    folder: Path,
    atoms: Atoms,
    structure_name: str,
    distortion_id: str,
    basesets: str,
    dft_block: str,
    cfg: dict,
) -> Path:
    sp_name = f"{structure_name}_{distortion_id}"

    distorted_cif = folder / f"{sp_name}.cif"
    write(distorted_cif, atoms)

    geo_txt = generate_crystal_block_from_atoms(atoms, sp_name)
    write_singlepoint_d12(folder, sp_name, geo_txt, basesets, dft_block)

    if cfg["PARTITION"] == "CPU_rune":
        write_sp_qsub_RUNE(folder, sp_name, cfg["NCORES"], cfg["RUNE"])
    elif cfg["PARTITION"] == "GPU_aq":
        write_sp_qsub_AQ(folder, sp_name, cfg["NCORES"])
    else:
        raise ValueError(f"Unknown partition: {cfg['PARTITION']}")

    return distorted_cif


def write_submit_script(out_root: Path) -> None:
    write_submit_all(out_root)