#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent

while not (PROJECT_ROOT / "util").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not find project root containing 'util'")
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
from util.config import load_yaml, normalize_config
from util.structure_io import (
    load_reference_atoms,
    apply_supercell_if_requested,
    report_structure,
)
from util.distance_checks import assert_structure_is_reasonable
from util.distortions import (
    build_mode_schedule,
    generate_valid_distortion,
    load_template_data,
    check_template_compatible,
    apply_template_distortion,
)
from util.metadata import write_json, write_run_hyperparams
from util.crystal_writers import (
    build_output_root,
    make_output_folder,
    prepare_crystal_blocks,
    write_singlepoint_package,
    write_submit_script,
)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distorted CRYSTAL single-point structures from YAML."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path. Relative paths are resolved against PROJECT_ROOT/configs.",
    )

    return parser.parse_args()


def resolve_config_path(config_path: Path) -> Path:
    if config_path.suffix not in {".yaml", ".yml"}:
        config_path = config_path.with_suffix(".yaml")

    if config_path.is_absolute():
        return config_path

    candidate_1 = PROJECT_ROOT / "configs" / config_path
    candidate_2 = PROJECT_ROOT / config_path

    if candidate_1.exists():
        return candidate_1

    if candidate_2.exists():
        return candidate_2

    return candidate_1


# ------------------------------------------------------------
# Structure selection
# ------------------------------------------------------------
def select_structure_dirs(cfg: dict) -> list[Path]:
    structures_dir = Path(cfg["STRUCTURES_DIR"])

    if not structures_dir.exists():
        raise FileNotFoundError(f"STRUCTURES_DIR does not exist: {structures_dir}")

    subdirs = [p for p in sorted(structures_dir.iterdir()) if p.is_dir()]

    only = cfg.get("ONLY")
    if only is not None:
        wanted = set(only)
        subdirs = [p for p in subdirs if p.name in wanted]

    if not subdirs:
        raise RuntimeError("No structure directories selected.")

    return subdirs


# ------------------------------------------------------------
# Distortion ID logic
# ------------------------------------------------------------
def make_distortion_id(cfg: dict, index: int, template_id: str | None = None) -> str:
    if cfg["DISTORTION_MODE"] == "template":
        if template_id is None:
            raise ValueError("template mode requires template_id")
        return str(template_id)

    return f"{cfg['BATCH']}{index:03d}"


# ------------------------------------------------------------
# Main per-structure workflow
# ------------------------------------------------------------
def generate_for_structure(structure_dir: Path, cfg: dict) -> None:
    structure_name = structure_dir.name

    print()
    print(f"[STRUCTURE] {structure_name}")

    atoms_ref, cif_file = load_reference_atoms(structure_dir, cfg)

    report_structure(atoms_ref, f"{structure_name} reference")
    assert_structure_is_reasonable(atoms_ref, f"{structure_name} reference")

    atoms_base, supercell_meta = apply_supercell_if_requested(atoms_ref, cfg)

    if cfg["SUPERCELL"] is not None:
        report_structure(atoms_base, f"{structure_name} supercell")
        assert_structure_is_reasonable(atoms_base, f"{structure_name} supercell")

    out_root = build_output_root(structure_dir, cfg)
    out_root.mkdir(parents=True, exist_ok=True)

    write_run_hyperparams(out_root, cfg, structure_name=structure_name)

    chem_comp, basesets, dft_block = prepare_crystal_blocks(structure_name, cfg)

    # Reproducibility:
    template_data = None

    if cfg["DISTORTION_MODE"] == "template":
        template_data = load_template_data(cfg["TEMPLATE_FILE"])
        check_template_compatible(atoms_base, template_data)

        n_available = len(template_data["ids"])
        if cfg["N_STRUCTURES"] > n_available:
            raise ValueError(
                f"Requested {cfg['N_STRUCTURES']} structures, "
                f"but template only contains {n_available}"
            )

    # total_random intentionally uses global np.random, matching generate_random_displ.py.
    if cfg["DISTORTION_MODE"] == "total_random":
        if cfg["RNG_SEED"] is None:
            raise ValueError("total_random requires rng_seed for reproducibility")
        np.random.seed(cfg["RNG_SEED"])
        rng = None
    else:
        rng = np.random.default_rng(cfg["RNG_SEED"])

    if cfg["DISTORTION_MODE"] == "scheduled_random":
        generation_items = build_mode_schedule(cfg["N_STRUCTURES"], cfg)
        rng.shuffle(generation_items)
    elif cfg["DISTORTION_MODE"] in {"total_random", "mixed_random"}:
        generation_items = [None] * cfg["N_STRUCTURES"]
    elif cfg["DISTORTION_MODE"] == "template":
        generation_items = list(range(cfg["N_STRUCTURES"]))
    else:
        raise ValueError(f"Unknown DISTORTION_MODE: {cfg['DISTORTION_MODE']}")

    summary = []
    total_attempts = 0

    for i, item in enumerate(generation_items):
        if cfg["DISTORTION_MODE"] == "template":
            distorted, meta = apply_template_distortion(
                atoms=atoms_base,
                template_data=template_data,
                template_index=item,
                cfg=cfg,
            )

            distortion_id = make_distortion_id(
                cfg,
                index=i,
                template_id=meta.get("template_id"),
            )
        else:
            scheduled_mode = item if cfg["DISTORTION_MODE"] == "scheduled_random" else None

            distorted, meta = generate_valid_distortion(
                atoms=atoms_base,
                cfg=cfg,
                rng=rng,
                scheduled_mode=scheduled_mode,
            )

            distortion_id = make_distortion_id(cfg, i)

        folder = make_output_folder(
            out_root,
            distortion_id=distortion_id,
            overwrite=cfg["OVERWRITE"],
        )

        distorted_cif = write_singlepoint_package(
            folder=folder,
            atoms=distorted,
            structure_name=structure_name,
            distortion_id=distortion_id,
            basesets=basesets,
            dft_block=dft_block,
            cfg=cfg,
        )

        meta.update(
            {
                "index": i,
                "batch": cfg["BATCH"],
                "legacy_charge": cfg.get("LEGACY_CHARGE"),
                "distortion_id": distortion_id,
                "folder_name": distortion_id,
                "structure": structure_name,
                "base_structure": chem_comp,
                "source_cif": str(cif_file),
                "sp_name": f"{structure_name}_{distortion_id}",
                "distorted_cif": str(distorted_cif),
                "natoms": len(distorted),
                "functional": cfg["FUNCTIONAL"],
                **supercell_meta,
            }
        )

        total_attempts += int(meta.get("generation_attempt", 1))

        write_json(folder / "distortion_meta.json", meta)
        summary.append(meta)

    write_json(out_root / "summary.json", summary)
    write_submit_script(out_root)

    avg_attempts = total_attempts / max(len(summary), 1)

    print(f"[DONE] {structure_name}")
    print(f"[OUT]  {out_root}")
    print(f"[INFO] Generated structures: {len(summary)}")
    print(f"[INFO] Average attempts / accepted distortion: {avg_attempts:.2f}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    args = parse_args()

    config_path = resolve_config_path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Config:       {config_path}")

    raw_cfg = load_yaml(config_path)
    cfg = normalize_config(raw_cfg, project_root=PROJECT_ROOT)

    if cfg["CIF_FILE"] is not None and cfg["ONLY"] is not None and len(cfg["ONLY"]) > 1:
        raise ValueError(
            "Explicit cif_file is only allowed for single-structure runs. "
            "Use only: [StructureName] or remove cif_file."
        )

    structure_dirs = select_structure_dirs(cfg)

    print(f"[INFO] Mode:        {cfg['DISTORTION_MODE']}")
    print(f"[INFO] Batch:       {cfg['BATCH']}")
    print(f"[INFO] Structures:  {[p.name for p in structure_dirs]}")

    for structure_dir in structure_dirs:
        try:
            generate_for_structure(structure_dir, cfg)
        except Exception as exc:
            print(f"[FAILED] {structure_dir.name}: {exc}")


if __name__ == "__main__":
    '''
    ./scripts/generate_distortions.py --config configs/AlN_010_total_random.yaml
    '''
    main()
