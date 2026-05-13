from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


SCRIPT_TO_MODE = {
    "generate_random_displ.py": "total_random",
    "gen_disto_struct.py": "mixed_random",
    "gen_dist_struct.py": "mixed_random",
    "mixed_disto_struct.py": "scheduled_random",
}

VALID_DISTORTION_MODES = {
    "total_random",
    "mixed_random",
    "scheduled_random",
    "template",
}


def find_project_root(start: Path | None = None) -> Path:
    root = Path(__file__).resolve() if start is None else Path(start).resolve()

    if root.is_file():
        root = root.parent

    while not (root / "util").exists():
        if root.parent == root:
            raise RuntimeError("Could not find project root containing 'util'")
        root = root.parent

    return root


def load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")

    return raw


def resolve_path(value: str | Path, base: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else base / p


def infer_distortion_mode(raw: dict[str, Any]) -> str:
    if raw.get("distortion_mode") is not None:
        mode = str(raw["distortion_mode"])
    elif raw.get("script") is not None:
        script = Path(str(raw["script"])).name
        if script not in SCRIPT_TO_MODE:
            raise ValueError(f"Unknown legacy script keyword: {script}")
        mode = SCRIPT_TO_MODE[script]
    else:
        raise KeyError(
            "Missing distortion_mode. New YAMLs should define e.g. "
            "distortion_mode: total_random"
        )

    if mode not in VALID_DISTORTION_MODES:
        raise ValueError(
            f"Invalid distortion_mode: {mode}. "
            f"Allowed: {sorted(VALID_DISTORTION_MODES)}"
        )

    return mode


def normalize_config(raw: dict[str, Any], project_root: Path) -> dict[str, Any]:
    mode = infer_distortion_mode(raw)

    root = resolve_path(raw.get("root", project_root), base=project_root)
    structures_dir = resolve_path(raw.get("structures_dir", "structures"), base=root)

    batch = raw.get("batch", raw.get("charge"))
    if batch is None:
        raise KeyError("Missing required key: batch")

    cfg: dict[str, Any] = {
        "ROOT": root,
        "STRUCTURES_DIR": structures_dir,
        "ONLY": raw.get("only"),
        "CIF_FILE": resolve_path(raw["cif_file"], base=root) if raw.get("cif_file") else None,

        "DISTORTION_MODE": mode,
        "LEGACY_SCRIPT": raw.get("script"),
        "BATCH": str(batch),
        "LEGACY_CHARGE": raw.get("charge"),

        "N_STRUCTURES": int(raw["n_structures"]),
        "RNG_SEED": None if raw.get("rng_seed") is None else int(raw["rng_seed"]),
        "SUPERCELL": raw.get("supercell"),
        "SUPERCELL_STAGE": "before_distortion",
        "OVERWRITE": bool(raw.get("overwrite", True)),

        "DISP_MAX": float(raw.get("disp_max", raw.get("max_disp", 0.0))),
        "STRAIN_MAX": float(raw.get("strain_max", raw.get("max_strain", 0.0))),

        "FRAC_SMALL_DISP": float(raw.get("frac_small_disp", 0.0)),
        "FRAC_MEDIUM_DISP": float(raw.get("frac_medium_disp", 0.0)),
        "FRAC_STRAIN_ONLY": float(raw.get("frac_strain_only", 0.0)),
        "FRAC_MIXED": float(raw.get("frac_mixed", 0.0)),

        "SMALL_DISP_MAX": float(raw.get("small_disp_max", 0.0)),
        "MEDIUM_DISP_MAX": float(raw.get("medium_disp_max", 0.0)),
        "SMALL_STRAIN_MAX": float(raw.get("small_strain_max", 0.0)),
        "MEDIUM_STRAIN_MAX": float(raw.get("medium_strain_max", 0.0)),

        "TEMPLATE_FILE": resolve_path(raw["template_file"], base=root) if raw.get("template_file") else None,
        "TEMPLATE_USE_CELL_DEFORMATION": bool(raw.get("template_use_cell_deformation", True)),
        "TEMPLATE_USE_FRAC_DISP": bool(raw.get("template_use_frac_disp", True)),
        "TEMPLATE_CELL_SCALE": float(raw.get("template_cell_scale", 1.0)),
        "TEMPLATE_ATOMIC_SCALE": float(raw.get("template_atomic_scale", 1.0)),

        "REJECT_ON_SHORT_DISTANCE": bool(raw.get("reject_on_short_distance", True)),
        "MAX_GENERATION_ATTEMPTS_PER_STRUCTURE": int(
            raw.get("max_generation_attempts_per_structure", 20000)
        ),

        "OUT_TAG": raw.get("out_tag"),
        "NCORES": int(raw.get("ncores", 5)),
        "PARTITION": raw.get("partition", "CPU_rune"),
        "RUNE": None if raw.get("rune") is None else int(raw["rune"]),

        "SHRINK": tuple(raw.get("shrink", [4, 4])),
        "MAXCYCLE": int(raw.get("maxcycle", 500)),
        "USE_GRADCAL": bool(raw.get("use_gradcal", True)),
        "SAVEWF": bool(raw.get("savewf", False)),
        "MULLIKEN": bool(raw.get("mulliken", False)),
        "FUNCTIONAL": raw.get("functional", "HSESol"),
    }

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    mode = cfg["DISTORTION_MODE"]

    if mode in {"total_random", "mixed_random"}:
        if cfg["DISP_MAX"] <= 0.0:
            raise ValueError(f"{mode} requires disp_max > 0")
        if cfg["STRAIN_MAX"] < 0.0:
            raise ValueError(f"{mode} requires strain_max >= 0")

    if mode == "scheduled_random":
        frac_sum = (
            cfg["FRAC_SMALL_DISP"]
            + cfg["FRAC_MEDIUM_DISP"]
            + cfg["FRAC_STRAIN_ONLY"]
            + cfg["FRAC_MIXED"]
        )
        if abs(frac_sum - 1.0) > 1e-8:
            raise ValueError(f"scheduled_random fractions must sum to 1.0, got {frac_sum}")

    if mode == "template" and cfg["TEMPLATE_FILE"] is None:
        raise KeyError("template mode requires template_file")
    