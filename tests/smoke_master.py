#!/usr/bin/env python3
"""
tests/smoke_master.py

Small smoke sweep runner.
Runs a tiny sweep using the normal run_master machinery.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = PROJECT_ROOT / "configs" / "master_cfg" / "smoke.yaml"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_master.py"),
        "--config",
        str(config),
        "--dry-run",
    ]

    print("Running smoke dry-run:")
    print(" ".join(map(str, cmd)))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
    