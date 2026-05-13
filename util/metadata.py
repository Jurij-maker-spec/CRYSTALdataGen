from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


def json_safe(obj: Any):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, tuple):
        return [json_safe(x) for x in obj]

    if isinstance(obj, list):
        return [json_safe(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    return obj


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.write_text(
        json.dumps(json_safe(data), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def collect_run_hyperparams(cfg: dict[str, Any], structure_name: str | None = None) -> dict:
    keys = [
        "ROOT",
        "STRUCTURES_DIR",
        "ONLY",
        "CIF_FILE",
        "DISTORTION_MODE",
        "LEGACY_SCRIPT",
        "BATCH",
        "LEGACY_CHARGE",
        "N_STRUCTURES",
        "RNG_SEED",
        "SUPERCELL",
        "SUPERCELL_STAGE",
        "OVERWRITE",
        "DISP_MAX",
        "STRAIN_MAX",
        "FRAC_SMALL_DISP",
        "FRAC_MEDIUM_DISP",
        "FRAC_STRAIN_ONLY",
        "FRAC_MIXED",
        "SMALL_DISP_MAX",
        "MEDIUM_DISP_MAX",
        "SMALL_STRAIN_MAX",
        "MEDIUM_STRAIN_MAX",
        "TEMPLATE_FILE",
        "TEMPLATE_USE_CELL_DEFORMATION",
        "TEMPLATE_USE_FRAC_DISP",
        "TEMPLATE_CELL_SCALE",
        "TEMPLATE_ATOMIC_SCALE",
        "REJECT_ON_SHORT_DISTANCE",
        "MAX_GENERATION_ATTEMPTS_PER_STRUCTURE",
        "OUT_TAG",
        "NCORES",
        "PARTITION",
        "RUNE",
        "SHRINK",
        "MAXCYCLE",
        "USE_GRADCAL",
        "SAVEWF",
        "MULLIKEN",
        "FUNCTIONAL",
    ]

    params = {"structure_name": structure_name}

    for key in keys:
        if key in cfg:
            params[key] = cfg[key]

    return json_safe(params)


def write_run_hyperparams(
    out_dir: Path,
    cfg: dict[str, Any],
    structure_name: str | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = collect_run_hyperparams(cfg, structure_name=structure_name)

    write_json(out_dir / "run_hyperparams.json", params)

    max_key_len = max(len(k) for k in params)
    lines = [f"{k:<{max_key_len}} : {v}" for k, v in sorted(params.items())]

    (out_dir / "run_hyperparams.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    