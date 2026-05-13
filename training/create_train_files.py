#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime
import argparse
import json
import random
import h5py
import yaml
import numpy as np
from ase import Atoms
from ase.io import write


# ============================================================
# CONFIG
# ============================================================

DEFAULT_PROJECT_ROOT = Path("/home/jha/jha/python_scripts/CRYSTALdataGen")
DEFAULT_CFG_ROOT = DEFAULT_PROJECT_ROOT / "configs" / "trainfile_cfg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/valid extxyz files from CRYSTALdataGen HDF5 dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file. Relative paths are resolved against configs/trainfile_cfg.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")

    return cfg


def resolve_path(value: str | Path, base: Path) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return base / p


def build_config(raw: dict[str, Any]) -> dict[str, Any]:
    project_root = resolve_path(
        raw.get("project_root", DEFAULT_PROJECT_ROOT),
        base=Path.cwd(),
    )

    return {
        "PROJECT_ROOT": project_root,
        "H5_PATH": resolve_path(raw.get("h5_path", "data/dataset.h5"), project_root),
        "EXPORT_METADATA_JSON": resolve_path(
            raw.get("export_metadata_json", "data/train_valid_metadata.json"),
            project_root,
        ),
        "TRAIN_XYZ": resolve_path(raw.get("train_xyz", "data/train.xyz"), project_root),
        "VALID_XYZ": resolve_path(raw.get("valid_xyz", "data/valid_interp.xyz"), project_root),

        "RNG_SEED": int(raw.get("rng_seed", 31415)),

        "VALID_SPLIT_MODE": raw.get("valid_split_mode", "fraction"),
        "VALID_PER_COMPOSITION": int(raw.get("valid_per_composition", 20)),
        "VALID_FRACTION_PER_COMPOSITION": float(raw.get("valid_fraction_per_composition", 0.10)),

        "INCLUDE_REFERENCES_IN_TRAIN": bool(raw.get("include_references_in_train", False)),
        "EXPORT_STRESS": bool(raw.get("export_stress", True)),
        "EXPORT_DIPOLES": bool(raw.get("export_dipoles", True)),
        "SHUFFLE_FINAL_LISTS": bool(raw.get("shuffle_final_lists", True)),

        "FILTER_BY_TRAINING_FLAG": bool(raw.get("filter_by_training_flag", True)),
        "FILTER_REFERENCES_BY_TRAINING_FLAG": bool(raw.get("filter_references_by_training_flag", False)),
        "ALLOW_SMALL_COMPOSITIONS": bool(raw.get("allow_small_compositions", True)),

        "STRUCTURE_SELECTION_MODE": raw.get("structure_selection_mode", "all"),
        "STRUCTURE_NAME": raw.get("structure_name"),
        "STRUCTURE_NAMES": raw.get("structure_names"),

        "USE_STRUCTURE_SUFFIX_IN_OUTPUT": bool(raw.get("use_structure_suffix_in_output", True)),

        "DISTORTION_ID_PREFIX": raw.get("distortion_id_prefix"),
        "DISTORTION_ID_PREFIXES": raw.get("distortion_id_prefixes"),
    }


def resolve_config_path(path: Path) -> Path:
    if path.suffix not in {".yaml", ".yml"}:
        path = path.with_suffix(".yaml")

    if path.is_absolute():
        return path

    return DEFAULT_CFG_ROOT / path


# ============================================================
# HELPERS
# ============================================================

def h5_string(obj):
    val = obj[()]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def group_has_training_flag_enabled(g):
    if "include_in_training" in g:
        include_flag = bool(np.array(g["include_in_training"]).item())
        if not include_flag:
            return False

    if "status" in g:
        status = h5_string(g["status"]).strip().lower()
        if status != "ok":
            return False

    return True


def normalized_distortion_prefixes(cfg: dict[str, Any]):
    prefixes = []

    if cfg["DISTORTION_ID_PREFIX"] is not None:
        prefixes.append(str(cfg["DISTORTION_ID_PREFIX"]))

    if cfg["DISTORTION_ID_PREFIXES"] is not None:
        prefixes.extend(str(x) for x in cfg["DISTORTION_ID_PREFIXES"])

    prefixes = sorted(set(prefixes))
    return None if len(prefixes) == 0 else prefixes


def distortion_id_matches_prefix(distortion_id: str, cfg: dict[str, Any]) -> bool:
    prefixes = normalized_distortion_prefixes(cfg)

    if prefixes is None:
        return True

    distortion_id = str(distortion_id)
    return any(distortion_id.startswith(prefix) for prefix in prefixes)


def selected_structure_names(h5, cfg: dict[str, Any]):
    if "structures" not in h5:
        raise KeyError("HDF5 file has no /structures group")

    available = sorted(h5["structures"].keys())
    mode = cfg["STRUCTURE_SELECTION_MODE"]

    if mode == "all":
        names = available

    elif mode == "one":
        if cfg["STRUCTURE_NAME"] is None:
            raise ValueError("structure_name must be set when structure_selection_mode='one'")
        names = [cfg["STRUCTURE_NAME"]]

    elif mode == "list":
        if cfg["STRUCTURE_NAMES"] is None:
            raise ValueError("structure_names must be set when structure_selection_mode='list'")
        names = list(cfg["STRUCTURE_NAMES"])

    else:
        raise ValueError("structure_selection_mode must be one of: 'all', 'one', 'list'")

    missing = [name for name in names if name not in h5["structures"]]
    if missing:
        raise KeyError(f"Requested structures not found in HDF5: {missing}")

    return sorted(names)


def output_paths(selected_names, cfg: dict[str, Any], natoms):
    if not cfg["USE_STRUCTURE_SUFFIX_IN_OUTPUT"]:
        return cfg["TRAIN_XYZ"], cfg["VALID_XYZ"]

    mode = cfg["STRUCTURE_SELECTION_MODE"]

    if mode == "all":
        return cfg["TRAIN_XYZ"], cfg["VALID_XYZ"]

    if mode == "one":
        suffix = cfg["STRUCTURE_NAME"]
    else:
        suffix = "_".join(selected_names)

    train_path = cfg["TRAIN_XYZ"].with_name(f"train_{suffix}_{natoms[0]}.xyz")
    valid_path = cfg["VALID_XYZ"].with_name(f"valid_{suffix}_{natoms[1]}.xyz")
    return train_path, valid_path


def load_atoms_from_group(g, cfg: dict[str, Any]):
    required = ["positions", "atomic_numbers", "lattice", "energy", "dft_forces"]
    missing = [key for key in required if key not in g]
    if missing:
        raise KeyError(f"Missing required datasets: {missing}")

    positions = np.array(g["positions"], dtype=float)
    numbers = np.array(g["atomic_numbers"], dtype=int)
    cell = np.array(g["lattice"], dtype=float)

    energy = float(np.array(g["energy"]))
    forces = np.array(g["dft_forces"], dtype=float)

    if forces.shape != (len(numbers), 3):
        raise ValueError(
            f"Forces shape mismatch: got {forces.shape}, expected {(len(numbers), 3)}"
        )

    pbc = np.array(g["pbc"]).astype(bool) if "pbc" in g else np.array([True, True, True], dtype=bool)

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    atoms.info["energy"] = energy
    atoms.arrays["forces"] = forces

    if cfg["EXPORT_STRESS"] and "stress" in g:
        stress = np.array(g["stress"], dtype=float)
        if stress.shape == (3, 3):
            atoms.info["stress"] = stress.reshape(-1)
        else:
            raise ValueError(f"Stress dataset has unexpected shape: {stress.shape}")

    if cfg["EXPORT_DIPOLES"]:
        if "dipole_au" in g:
            dip = np.array(g["dipole_au"], dtype=float)
            if dip.shape == (3,):
                atoms.info["dipole_au_x"] = float(dip[0])
                atoms.info["dipole_au_y"] = float(dip[1])
                atoms.info["dipole_au_z"] = float(dip[2])

        if "dipole_debye" in g:
            dip = np.array(g["dipole_debye"], dtype=float)
            if dip.shape == (3,):
                atoms.info["dipole_debye_x"] = float(dip[0])
                atoms.info["dipole_debye_y"] = float(dip[1])
                atoms.info["dipole_debye_z"] = float(dip[2])

    if "status" in g:
        atoms.info["status"] = h5_string(g["status"])

    if "include_in_training" in g:
        atoms.info["include_in_training"] = int(np.array(g["include_in_training"]).item())

    return atoms


def collect_reference_atoms(h5, selected_names, cfg: dict[str, Any]):
    refs = []
    n_skipped = 0

    structures = h5["structures"]

    for structure_name in selected_names:
        structure_group = structures[structure_name]

        if "reference" not in structure_group:
            print(f"[WARN] Missing reference group for {structure_name}")
            n_skipped += 1
            continue

        ref_group = structure_group["reference"]

        if cfg["FILTER_REFERENCES_BY_TRAINING_FLAG"] and not group_has_training_flag_enabled(ref_group):
            print(f"[SKIP] Reference {structure_name}: flagged as not for training")
            n_skipped += 1
            continue

        try:
            atoms = load_atoms_from_group(ref_group, cfg)
        except Exception as e:
            print(f"[WARN] Skipping reference {structure_name}: {e}")
            n_skipped += 1
            continue

        atoms.info["config_type"] = "reference"
        atoms.info["structure_name"] = structure_name
        atoms.info["distortion_id"] = "reference"

        refs.append(atoms)

    return refs, n_skipped


def collect_distortion_atoms(h5, selected_names, cfg: dict[str, Any]):
    grouped = {}
    skipped_counts = {}
    skipped_prefix_counts = {}
    skipped_explicit = []

    structures = h5["structures"]

    for structure_name in selected_names:
        structure_group = structures[structure_name]

        if "distortions" not in structure_group:
            grouped[structure_name] = []
            skipped_counts[structure_name] = 0
            skipped_prefix_counts[structure_name] = 0
            continue

        distortions_group = structure_group["distortions"]
        grouped[structure_name] = []
        skipped_counts[structure_name] = 0
        skipped_prefix_counts[structure_name] = 0

        for distortion_id in sorted(distortions_group.keys()):
            g = distortions_group[distortion_id]

            if not distortion_id_matches_prefix(distortion_id, cfg):
                skipped_prefix_counts[structure_name] += 1
                continue

            if cfg["FILTER_BY_TRAINING_FLAG"] and not group_has_training_flag_enabled(g):
                skipped_counts[structure_name] += 1
                skipped_explicit.append(distortion_id)
                continue

            try:
                atoms = load_atoms_from_group(g, cfg)
            except Exception as e:
                print(f"[WARN] Skipping {structure_name}/{distortion_id}: {e}")
                skipped_counts[structure_name] += 1
                continue

            atoms.info["config_type"] = "distortion"
            atoms.info["structure_name"] = structure_name
            atoms.info["distortion_id"] = distortion_id

            grouped[structure_name].append(atoms)

    return grouped, skipped_counts, skipped_prefix_counts, skipped_explicit


def split_grouped_distortions(grouped, cfg: dict[str, Any]):
    rng = random.Random(cfg["RNG_SEED"])

    train_atoms = []
    valid_atoms = []

    split_mode = cfg["VALID_SPLIT_MODE"]
    valid_per_composition = cfg["VALID_PER_COMPOSITION"]
    valid_fraction_per_composition = cfg["VALID_FRACTION_PER_COMPOSITION"]

    if split_mode not in {"count", "fraction"}:
        raise ValueError("valid_split_mode must be 'count' or 'fraction'")

    if split_mode == "fraction":
        if not (0.0 <= valid_fraction_per_composition < 1.0):
            raise ValueError("valid_fraction_per_composition must satisfy 0.0 <= fraction < 1.0")

    for structure_name in sorted(grouped.keys()):
        atoms_list = list(grouped[structure_name])
        n_total = len(atoms_list)

        if n_total == 0:
            print(f"[WARN] No usable distortions for {structure_name}")
            continue

        if n_total == 1:
            train_atoms.extend(atoms_list)
            print(f"[OK] {structure_name}: train distortions = 1, valid distortions = 0")
            continue

        if split_mode == "count":
            if valid_per_composition >= n_total:
                if cfg["ALLOW_SMALL_COMPOSITIONS"]:
                    n_valid_target = max(1, n_total - 1)
                else:
                    raise ValueError(
                        f"{structure_name}: valid_per_composition={valid_per_composition} "
                        f"is >= number of distortions ({n_total})"
                    )
            else:
                n_valid_target = valid_per_composition

        else:
            raw_n_valid = int(round(valid_fraction_per_composition * n_total))

            if raw_n_valid <= 0:
                n_valid_target = 1 if valid_fraction_per_composition > 0.0 else 0
            else:
                n_valid_target = raw_n_valid

            if n_valid_target >= n_total:
                if cfg["ALLOW_SMALL_COMPOSITIONS"]:
                    n_valid_target = max(1, n_total - 1)
                else:
                    raise ValueError(
                        f"{structure_name}: fraction-based split would place all "
                        f"{n_total} distortions into validation"
                    )

        indices = list(range(n_total))
        rng.shuffle(indices)
        valid_idx = set(indices[:n_valid_target])

        n_train = 0
        n_valid = 0

        for i, atoms in enumerate(atoms_list):
            if i in valid_idx:
                valid_atoms.append(atoms)
                n_valid += 1
            else:
                train_atoms.append(atoms)
                n_train += 1

        if split_mode == "count":
            split_desc = f"mode=count, target_valid={valid_per_composition}"
        else:
            split_desc = (
                f"mode=fraction, fraction={valid_fraction_per_composition:.4f}, "
                f"target_valid={n_valid_target}"
            )

        print(
            f"[OK] {structure_name}: train distortions = {n_train}, "
            f"valid distortions = {n_valid} ({split_desc}, total={n_total})"
        )

    return train_atoms, valid_atoms


def write_xyz(path: Path, atoms_list):
    if not atoms_list:
        print(f"[WARN] No structures to write for {path}")
        return
    write(path, atoms_list, format="extxyz")
    print(f"[OK] Wrote {len(atoms_list)} structures to {path}")


def summarize_split(
    train_atoms,
    valid_atoms,
    cfg: dict[str, Any],
    skipped_distortions=None,
    skipped_distortions_exp=None,
    skipped_references=0,
    skipped_prefix_distortions=None,
):
    train_by_type = {}
    valid_by_type = {}

    for atoms in train_atoms:
        name = atoms.info.get("structure_name", "UNKNOWN")
        train_by_type[name] = train_by_type.get(name, 0) + 1

    for atoms in valid_atoms:
        name = atoms.info.get("structure_name", "UNKNOWN")
        valid_by_type[name] = valid_by_type.get(name, 0) + 1

    print("\n=== Split summary ===")
    print(f"Train total                : {len(train_atoms)}")
    print(f"Valid total                : {len(valid_atoms)}")
    print(f"Skipped references         : {skipped_references}")

    total_skipped_dist = 0 if skipped_distortions is None else sum(skipped_distortions.values())
    print(f"Skipped distortions        : {total_skipped_dist}")

    if skipped_distortions_exp is not None:
        for dist in skipped_distortions_exp:
            print(f"--->    : {dist}")

    total_skipped_prefix = 0 if skipped_prefix_distortions is None else sum(skipped_prefix_distortions.values())
    print(f"Skipped by prefix filter   : {total_skipped_prefix}")

    prefixes = normalized_distortion_prefixes(cfg)
    print(f"Distortion prefix filter   : {prefixes if prefixes is not None else 'None'}")

    print("\nPer structure:")
    all_names = sorted(
        set(train_by_type)
        | set(valid_by_type)
        | set((skipped_distortions or {}).keys())
        | set((skipped_prefix_distortions or {}).keys())
    )

    for name in all_names:
        print(
            f"  {name:12s} "
            f"train={train_by_type.get(name, 0):4d}  "
            f"valid={valid_by_type.get(name, 0):4d}  "
            f"skipped={(skipped_distortions or {}).get(name, 0):4d}  "
            f"prefix_skip={(skipped_prefix_distortions or {}).get(name, 0):4d}"
        )


def metadata_json_path(selected_names, cfg: dict[str, Any]):
    if not cfg["USE_STRUCTURE_SUFFIX_IN_OUTPUT"]:
        return cfg["EXPORT_METADATA_JSON"]

    mode = cfg["STRUCTURE_SELECTION_MODE"]

    if mode == "all":
        return cfg["EXPORT_METADATA_JSON"]

    if mode == "one":
        suffix = cfg["STRUCTURE_NAME"]
    else:
        suffix = "_".join(selected_names)

    return cfg["EXPORT_METADATA_JSON"].with_name(f"tv_meta_{suffix}.json")


def write_export_metadata_json(
    path: Path,
    selected_names,
    train_path: Path,
    valid_path: Path,
    train_atoms,
    valid_atoms,
    cfg: dict[str, Any],
    skipped_references=0,
    skipped_distortions=None,
    skipped_prefix_distortions=None,
    skipped_distortions_explicit=None,
):
    prefixes = normalized_distortion_prefixes(cfg)

    train_by_structure = {}
    valid_by_structure = {}

    for atoms in train_atoms:
        name = atoms.info.get("structure_name", "UNKNOWN")
        train_by_structure[name] = train_by_structure.get(name, 0) + 1

    for atoms in valid_atoms:
        name = atoms.info.get("structure_name", "UNKNOWN")
        valid_by_structure[name] = valid_by_structure.get(name, 0) + 1

    all_names = sorted(
        set(selected_names)
        | set(train_by_structure.keys())
        | set(valid_by_structure.keys())
        | set((skipped_distortions or {}).keys())
        | set((skipped_prefix_distortions or {}).keys())
    )

    per_structure = {}
    for name in all_names:
        per_structure[name] = {
            "train_count": int(train_by_structure.get(name, 0)),
            "valid_count": int(valid_by_structure.get(name, 0)),
            "skipped_count": int((skipped_distortions or {}).get(name, 0)),
            "prefix_skipped_count": int((skipped_prefix_distortions or {}).get(name, 0)),
        }

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(cfg["PROJECT_ROOT"]),
        "h5_path": str(cfg["H5_PATH"]),
        "train_xyz": str(train_path),
        "valid_xyz": str(valid_path),
        "settings": {
            "rng_seed": int(cfg["RNG_SEED"]),
            "valid_split_mode": str(cfg["VALID_SPLIT_MODE"]),
            "valid_per_composition": int(cfg["VALID_PER_COMPOSITION"]),
            "valid_fraction_per_composition": float(cfg["VALID_FRACTION_PER_COMPOSITION"]),
            "include_references_in_train": bool(cfg["INCLUDE_REFERENCES_IN_TRAIN"]),
            "export_stress": bool(cfg["EXPORT_STRESS"]),
            "export_dipoles": bool(cfg["EXPORT_DIPOLES"]),
            "shuffle_final_lists": bool(cfg["SHUFFLE_FINAL_LISTS"]),
            "filter_by_training_flag": bool(cfg["FILTER_BY_TRAINING_FLAG"]),
            "filter_references_by_training_flag": bool(cfg["FILTER_REFERENCES_BY_TRAINING_FLAG"]),
            "allow_small_compositions": bool(cfg["ALLOW_SMALL_COMPOSITIONS"]),
            "structure_selection_mode": str(cfg["STRUCTURE_SELECTION_MODE"]),
            "structure_name": None if cfg["STRUCTURE_SELECTION_MODE"] != "one" else str(cfg["STRUCTURE_NAME"]),
            "distortion_id_prefix": None if cfg["DISTORTION_ID_PREFIX"] is None else str(cfg["DISTORTION_ID_PREFIX"]),
            "distortion_id_prefixes": (
                None if cfg["DISTORTION_ID_PREFIXES"] is None
                else [str(x) for x in cfg["DISTORTION_ID_PREFIXES"]]
            ),
            "resolved_distortion_prefixes": prefixes,
            "use_structure_suffix_in_output": bool(cfg["USE_STRUCTURE_SUFFIX_IN_OUTPUT"]),
        },
        "selected_structures": [str(x) for x in selected_names],
        "summary": {
            "n_train_total": int(len(train_atoms)),
            "n_valid_total": int(len(valid_atoms)),
            "n_reference_train": int(sum(1 for a in train_atoms if a.info.get("config_type") == "reference")),
            "n_distortion_train": int(sum(1 for a in train_atoms if a.info.get("config_type") == "distortion")),
            "n_distortion_valid": int(sum(1 for a in valid_atoms if a.info.get("config_type") == "distortion")),
            "n_skipped_references": int(skipped_references),
            "n_skipped_distortions": int(sum((skipped_distortions or {}).values())),
            "n_skipped_by_prefix": int(sum((skipped_prefix_distortions or {}).values())),
            "skipped_distortion_ids_explicit": (
                [] if skipped_distortions_explicit is None
                else [str(x) for x in skipped_distortions_explicit]
            ),
        },
        "per_structure": per_structure,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=False)

    print(f"[OK] Wrote export metadata JSON to {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    raw_cfg = load_yaml(config_path)
    cfg = build_config(raw_cfg)

    if not cfg["H5_PATH"].exists():
        raise FileNotFoundError(f"Dataset not found: {cfg['H5_PATH']}")

    with h5py.File(cfg["H5_PATH"], "r") as h5:
        selected_names = selected_structure_names(h5, cfg)

        print("Selected structures:")
        for name in selected_names:
            print(f"  - {name}")

        if cfg["INCLUDE_REFERENCES_IN_TRAIN"]:
            reference_atoms, skipped_references = collect_reference_atoms(h5, selected_names, cfg)
        else:
            reference_atoms, skipped_references = [], 0

        grouped_distortions, skipped_distortions, skipped_prefix_distortions, skipped_explicit = (
            collect_distortion_atoms(h5, selected_names, cfg)
        )

    train_dist_atoms, valid_atoms = split_grouped_distortions(
        grouped=grouped_distortions,
        cfg=cfg,
    )

    train_atoms = reference_atoms + train_dist_atoms

    if cfg["SHUFFLE_FINAL_LISTS"]:
        rng = random.Random(cfg["RNG_SEED"])
        rng.shuffle(train_atoms)
        rng.shuffle(valid_atoms)


    n_train, n_valid = len(train_atoms), len(valid_atoms)
    train_path, valid_path = output_paths(selected_names, cfg, (n_train, n_valid))

    print(f"\nEXPORT_STRESS = {cfg['EXPORT_STRESS']}")
    print(f"EXPORT_DIPOLES = {cfg['EXPORT_DIPOLES']}")
    print(f"FILTER_BY_TRAINING_FLAG = {cfg['FILTER_BY_TRAINING_FLAG']}")
    print(f"VALID_SPLIT_MODE = {cfg['VALID_SPLIT_MODE']}")
    print(f"VALID_PER_COMPOSITION = {cfg['VALID_PER_COMPOSITION']}")
    print(f"VALID_FRACTION_PER_COMPOSITION = {cfg['VALID_FRACTION_PER_COMPOSITION']}")

    summarize_split(
        train_atoms,
        valid_atoms,
        cfg=cfg,
        skipped_distortions=skipped_distortions,
        skipped_distortions_exp=skipped_explicit,
        skipped_references=skipped_references,
        skipped_prefix_distortions=skipped_prefix_distortions,
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    write_xyz(train_path, train_atoms)
    write_xyz(valid_path, valid_atoms)

    metadata_json = metadata_json_path(selected_names, cfg)
    write_export_metadata_json(
        path=metadata_json,
        selected_names=selected_names,
        train_path=train_path,
        valid_path=valid_path,
        train_atoms=train_atoms,
        valid_atoms=valid_atoms,
        cfg=cfg,
        skipped_references=skipped_references,
        skipped_distortions=skipped_distortions,
        skipped_prefix_distortions=skipped_prefix_distortions,
        skipped_distortions_explicit=skipped_explicit,
    )


if __name__ == "__main__":
    main()
