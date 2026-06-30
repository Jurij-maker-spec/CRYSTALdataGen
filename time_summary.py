#!/usr/bin/env python3
"""
time_summary.py

Summarize consumed CRYSTAL CPU core-hours and MACE/MACELES GPU wall-hours.

Default behavior from the project root:
    python time_summary.py

Useful variants:
    python time_summary.py --project-root /path/to/CRYSTALdataGen
    python time_summary.py --output-dir time_summary
    python time_summary.py --gap-hours 6
    python time_summary.py --include-slurm-out

Outputs:
    time_summary/time_summary.txt
    time_summary/time_summary.json
    time_summary/time_summary_grouped.csv
    time_summary/crystal_cpu_detail.csv
    time_summary/gpu_training_detail.csv

Notes:
- CRYSTAL CPU time is parsed as CPU core-seconds/core-hours.
  The preferred source is the final "TOTAL CPU TIME = ..." line.
- GPU training time is estimated from train.log timestamps.
  With one GPU per run, GPU-hours = training wall-hours.
- By default, large gaps between consecutive train.log timestamps are subtracted
  from the GPU total. This avoids strong overcounting for appended/resumed logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"

TOTAL_CPU_RE = re.compile(
    rf"TOTAL\s+CPU\s+TIME\s*=\s*(?P<seconds>{FLOAT_RE})",
    re.IGNORECASE,
)
NODE_CPU_RE = re.compile(
    rf"NODE\s+\d+\s+CPU\s+TIME\s*=\s*(?P<seconds>{FLOAT_RE})",
    re.IGNORECASE,
)
END_TCPU_RE = re.compile(
    rf"\bEND\b.*?\bTELAPSE\s+(?P<telapse>{FLOAT_RE})\s+TCPU\s+(?P<tcpu>{FLOAT_RE})",
    re.IGNORECASE,
)
ANY_TCPU_RE = re.compile(
    rf"\bTELAPSE\s+(?P<telapse>{FLOAT_RE})\s+TCPU\s+(?P<tcpu>{FLOAT_RE})",
    re.IGNORECASE,
)
LOG_TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[\.,]\d{1,6})?)\s"
)


@dataclass
class CrystalCPURecord:
    source_type: str
    path: str
    root_dir: str
    structure: str
    functional: str
    job_kind: str
    calculation_id: str
    cpu_seconds: Optional[float]
    cpu_hours: Optional[float]
    parse_source: str
    n_node_lines: int
    wall_seconds_from_end_line: Optional[float]
    status: str
    error: str = ""


@dataclass
class GPUTrainingRecord:
    source_type: str
    path: str
    dataset_split: str
    sweep_id: str
    run_id: str
    structure: str
    run_structure: str
    functional: str
    first_timestamp: str
    last_timestamp: str
    n_timestamps: int
    raw_wall_hours: Optional[float]
    adjusted_wall_hours: Optional[float]
    ignored_gap_hours: Optional[float]
    n_large_gaps: int
    n_segments: int
    gpus_per_run: int
    gpu_hours: Optional[float]
    status: str
    error: str = ""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def parse_crystal_cpu_seconds(text: str) -> tuple[Optional[float], str, int, Optional[float]]:
    """
    Return (cpu_seconds, parse_source, n_node_lines, wall_seconds_from_end_line).

    Preference order:
    1. Final TOTAL CPU TIME line. This is the summed CPU/core time over MPI ranks/nodes.
    2. Sum of NODE ... CPU TIME lines, if no TOTAL line exists.
    3. Final END ... TELAPSE ... TCPU line.
    4. Last generic TELAPSE ... TCPU line.
    """
    total_matches = [parse_float(m.group("seconds")) for m in TOTAL_CPU_RE.finditer(text)]
    node_matches = [parse_float(m.group("seconds")) for m in NODE_CPU_RE.finditer(text)]
    end_matches = list(END_TCPU_RE.finditer(text))
    any_tcpu_matches = list(ANY_TCPU_RE.finditer(text))

    wall_seconds_from_end_line = None
    if end_matches:
        wall_seconds_from_end_line = parse_float(end_matches[-1].group("telapse"))

    if total_matches:
        return total_matches[-1], "TOTAL CPU TIME", len(node_matches), wall_seconds_from_end_line

    if node_matches:
        return sum(node_matches), "sum NODE CPU TIME", len(node_matches), wall_seconds_from_end_line

    if end_matches:
        return parse_float(end_matches[-1].group("tcpu")), "END TCPU", len(node_matches), wall_seconds_from_end_line

    if any_tcpu_matches:
        last = any_tcpu_matches[-1]
        return parse_float(last.group("tcpu")), "last TCPU", len(node_matches), parse_float(last.group("telapse"))

    return None, "missing", len(node_matches), wall_seconds_from_end_line


def parse_train_log_timestamps(path: Path) -> list[datetime]:
    timestamps: list[datetime] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = LOG_TIMESTAMP_RE.match(line)
            if not match:
                continue
            stamp = match.group("timestamp").replace(",", ".")
            try:
                timestamps.append(datetime.fromisoformat(stamp))
            except ValueError:
                continue
    return timestamps


def summarize_timestamp_span(
    timestamps: list[datetime], gap_hours: float
) -> tuple[Optional[float], Optional[float], Optional[float], int, int, str, str]:
    """
    Return raw_wall_hours, adjusted_wall_hours, ignored_gap_hours,
    n_large_gaps, n_segments, status, error.
    """
    if not timestamps:
        return None, None, None, 0, 0, "failed", "no parseable timestamps"

    if len(timestamps) == 1:
        return 0.0, 0.0, 0.0, 0, 1, "one_timestamp", "only one parseable timestamp"

    timestamps = sorted(timestamps)
    raw_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    threshold_seconds = gap_hours * 3600.0

    adjusted_seconds = 0.0
    ignored_gap_seconds = 0.0
    n_large_gaps = 0

    for prev, cur in zip(timestamps[:-1], timestamps[1:]):
        delta = (cur - prev).total_seconds()
        if delta < 0:
            continue
        if delta > threshold_seconds:
            ignored_gap_seconds += delta
            n_large_gaps += 1
        else:
            adjusted_seconds += delta

    n_segments = n_large_gaps + 1
    status = "ok"
    if n_large_gaps:
        status = "ok_gap_adjusted"

    return (
        raw_seconds / 3600.0,
        adjusted_seconds / 3600.0,
        ignored_gap_seconds / 3600.0,
        n_large_gaps,
        n_segments,
        status,
        "",
    )


def should_skip_out_file(path: Path, include_slurm_out: bool) -> bool:
    name = path.name.lower()
    if not include_slurm_out and name.startswith("slurm-") and name.endswith(".out"):
        return True
    return False


def infer_functional_from_path(path: Path, root: Optional[Path] = None) -> str:
    joined = "/".join(path.parts).lower()
    root_name = root.name.lower() if root is not None else ""
    if "pbe" in root_name or "structures_pbe" in joined or "_pbe" in joined or "/pbe/" in joined:
        return "PBEsol"
    return "HSEsol"


def infer_job_kind(path: Path) -> str:
    name = path.name.lower()
    if "freq" in name:
        return "freq"
    if "_sp" in name or name.endswith("sp.out"):
        return "sp"
    if "geoopt" in name or "opt" in name:
        return "geoopt"
    return "unknown"


def list_known_structures(structure_roots: Iterable[Path]) -> list[str]:
    names: set[str] = set()
    for root in structure_roots:
        if root.exists():
            for child in root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    names.add(child.name)
    return sorted(names, key=len, reverse=True)


def infer_structure_from_name(name: str, known_structures: Iterable[str]) -> str:
    # Prefer exact known structure prefixes, longest first. This handles TiO2_rutil before TiO2.
    for struct in known_structures:
        if name == struct or name.startswith(struct + "_") or name.startswith(struct + "-"):
            return struct
        if name.endswith("_" + struct):
            return struct
    # Conservative fallback: strip common trailing split markers.
    parts = name.split("_")
    if len(parts) >= 2 and parts[1] == "rutil":
        return "_".join(parts[:2])
    return parts[0] if parts else "unknown"


def infer_run_structure(run_id: str, known_structures: Iterable[str]) -> str:
    for struct in known_structures:
        if run_id.endswith("_" + struct) or f"_{struct}_" in run_id:
            return struct
    return infer_structure_from_name(run_id, known_structures)


def collect_crystal_cpu_records(
    structure_roots: list[Path],
    known_structures: list[str],
    include_slurm_out: bool = False,
) -> list[CrystalCPURecord]:
    records: list[CrystalCPURecord] = []

    for root in structure_roots:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*.out")):
            if should_skip_out_file(path, include_slurm_out=include_slurm_out):
                continue

            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path

            structure = rel.parts[0] if len(rel.parts) > 1 else infer_structure_from_name(path.name, known_structures)
            functional = infer_functional_from_path(path, root)
            job_kind = infer_job_kind(path)
            calculation_id = path.parent.name

            try:
                text = read_text(path)
                cpu_seconds, parse_source, n_node_lines, wall_seconds = parse_crystal_cpu_seconds(text)
                if cpu_seconds is None:
                    records.append(
                        CrystalCPURecord(
                            source_type="crystal_cpu",
                            path=str(path),
                            root_dir=str(root),
                            structure=structure,
                            functional=functional,
                            job_kind=job_kind,
                            calculation_id=calculation_id,
                            cpu_seconds=None,
                            cpu_hours=None,
                            parse_source=parse_source,
                            n_node_lines=n_node_lines,
                            wall_seconds_from_end_line=wall_seconds,
                            status="failed",
                            error="no CPU time pattern found",
                        )
                    )
                    continue

                records.append(
                    CrystalCPURecord(
                        source_type="crystal_cpu",
                        path=str(path),
                        root_dir=str(root),
                        structure=structure,
                        functional=functional,
                        job_kind=job_kind,
                        calculation_id=calculation_id,
                        cpu_seconds=cpu_seconds,
                        cpu_hours=cpu_seconds / 3600.0,
                        parse_source=parse_source,
                        n_node_lines=n_node_lines,
                        wall_seconds_from_end_line=wall_seconds,
                        status="ok",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - robust summary script, record and continue
                records.append(
                    CrystalCPURecord(
                        source_type="crystal_cpu",
                        path=str(path),
                        root_dir=str(root),
                        structure=structure,
                        functional=functional,
                        job_kind=job_kind,
                        calculation_id=calculation_id,
                        cpu_seconds=None,
                        cpu_hours=None,
                        parse_source="exception",
                        n_node_lines=0,
                        wall_seconds_from_end_line=None,
                        status="failed",
                        error=str(exc),
                    )
                )

    return records


def collect_gpu_training_records(
    results_root: Path,
    known_structures: list[str],
    gap_hours: float,
    gpus_per_run: int,
) -> list[GPUTrainingRecord]:
    records: list[GPUTrainingRecord] = []

    if not results_root.exists():
        return records

    for path in sorted(results_root.rglob("train.log")):
        try:
            rel = path.relative_to(results_root)
            parts = rel.parts
            dataset_split = parts[0] if len(parts) >= 1 else "unknown"
            sweep_id = parts[1] if len(parts) >= 2 else "unknown"
            run_id = parts[2] if len(parts) >= 3 else path.parent.name
            structure = infer_structure_from_name(dataset_split, known_structures)
            run_structure = infer_run_structure(run_id, known_structures)
            functional = "PBEsol" if "pbe" in dataset_split.lower() or "pbe" in sweep_id.lower() else "HSEsol"

            timestamps = parse_train_log_timestamps(path)
            (
                raw_wall_hours,
                adjusted_wall_hours,
                ignored_gap_hours,
                n_large_gaps,
                n_segments,
                status,
                error,
            ) = summarize_timestamp_span(timestamps, gap_hours=gap_hours)

            gpu_hours = None
            if adjusted_wall_hours is not None:
                gpu_hours = adjusted_wall_hours * gpus_per_run

            first_timestamp = timestamps[0].isoformat(sep=" ") if timestamps else ""
            last_timestamp = timestamps[-1].isoformat(sep=" ") if timestamps else ""

            records.append(
                GPUTrainingRecord(
                    source_type="gpu_training",
                    path=str(path),
                    dataset_split=dataset_split,
                    sweep_id=sweep_id,
                    run_id=run_id,
                    structure=structure,
                    run_structure=run_structure,
                    functional=functional,
                    first_timestamp=first_timestamp,
                    last_timestamp=last_timestamp,
                    n_timestamps=len(timestamps),
                    raw_wall_hours=raw_wall_hours,
                    adjusted_wall_hours=adjusted_wall_hours,
                    ignored_gap_hours=ignored_gap_hours,
                    n_large_gaps=n_large_gaps,
                    n_segments=n_segments,
                    gpus_per_run=gpus_per_run,
                    gpu_hours=gpu_hours,
                    status=status,
                    error=error,
                )
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                GPUTrainingRecord(
                    source_type="gpu_training",
                    path=str(path),
                    dataset_split="unknown",
                    sweep_id="unknown",
                    run_id=path.parent.name,
                    structure="unknown",
                    run_structure="unknown",
                    functional="unknown",
                    first_timestamp="",
                    last_timestamp="",
                    n_timestamps=0,
                    raw_wall_hours=None,
                    adjusted_wall_hours=None,
                    ignored_gap_hours=None,
                    n_large_gaps=0,
                    n_segments=0,
                    gpus_per_run=gpus_per_run,
                    gpu_hours=None,
                    status="failed",
                    error=str(exc),
                )
            )

    return records


def sum_optional(values: Iterable[Optional[float]]) -> float:
    return sum(v for v in values if v is not None)


def count_status(records: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[getattr(rec, "status", "unknown")] += 1
    return dict(sorted(counts.items()))


def group_crystal(records: list[CrystalCPURecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for rec in records:
        key = (rec.structure, rec.functional, rec.job_kind)
        if key not in grouped:
            grouped[key] = {
                "source_type": "crystal_cpu",
                "structure": rec.structure,
                "functional": rec.functional,
                "job_kind": rec.job_kind,
                "dataset_split": "",
                "sweep_id": "",
                "n_files": 0,
                "n_ok": 0,
                "n_failed": 0,
                "cpu_hours": 0.0,
                "gpu_hours": 0.0,
                "raw_wall_hours": 0.0,
                "ignored_gap_hours": 0.0,
            }
        row = grouped[key]
        row["n_files"] += 1
        if rec.cpu_hours is not None:
            row["n_ok"] += 1
            row["cpu_hours"] += rec.cpu_hours
        else:
            row["n_failed"] += 1
    return list(grouped.values())


def group_gpu(records: list[GPUTrainingRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for rec in records:
        key = (rec.structure, rec.functional, rec.dataset_split, rec.sweep_id)
        if key not in grouped:
            grouped[key] = {
                "source_type": "gpu_training",
                "structure": rec.structure,
                "functional": rec.functional,
                "job_kind": "training",
                "dataset_split": rec.dataset_split,
                "sweep_id": rec.sweep_id,
                "n_files": 0,
                "n_ok": 0,
                "n_failed": 0,
                "cpu_hours": 0.0,
                "gpu_hours": 0.0,
                "raw_wall_hours": 0.0,
                "ignored_gap_hours": 0.0,
            }
        row = grouped[key]
        row["n_files"] += 1
        if rec.gpu_hours is not None:
            row["n_ok"] += 1
            row["gpu_hours"] += rec.gpu_hours
            row["raw_wall_hours"] += rec.raw_wall_hours or 0.0
            row["ignored_gap_hours"] += rec.ignored_gap_hours or 0.0
        else:
            row["n_failed"] += 1
    return list(grouped.values())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_hours(hours: float) -> str:
    return f"{hours:,.2f}"


def make_text_summary(
    crystal_records: list[CrystalCPURecord],
    gpu_records: list[GPUTrainingRecord],
    grouped_rows: list[dict[str, object]],
    project_root: Path,
    structure_roots: list[Path],
    results_root: Path,
    gap_hours: float,
    gpus_per_run: int,
) -> str:
    total_cpu_hours = sum_optional(rec.cpu_hours for rec in crystal_records)
    total_gpu_hours = sum_optional(rec.gpu_hours for rec in gpu_records)
    total_gpu_raw_wall_hours = sum_optional(rec.raw_wall_hours for rec in gpu_records)
    total_ignored_gap_hours = sum_optional(rec.ignored_gap_hours for rec in gpu_records)

    n_crystal_ok = sum(1 for rec in crystal_records if rec.cpu_hours is not None)
    n_crystal_failed = len(crystal_records) - n_crystal_ok
    n_gpu_ok = sum(1 for rec in gpu_records if rec.gpu_hours is not None)
    n_gpu_failed = len(gpu_records) - n_gpu_ok

    lines: list[str] = []
    lines.append("TIME SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Project root              : {project_root}")
    lines.append(f"CRYSTAL roots             : {', '.join(str(p) for p in structure_roots if p.exists()) or 'none found'}")
    lines.append(f"Results root              : {results_root if results_root.exists() else 'not found'}")
    lines.append(f"GPU count per train.log   : {gpus_per_run}")
    lines.append(f"Resume gap threshold      : {gap_hours:g} h")
    lines.append("")
    lines.append("TOTALS")
    lines.append("-" * 80)
    lines.append(f"CRYSTAL CPU core-hours    : {format_hours(total_cpu_hours)}")
    lines.append(f"MACE GPU-hours adjusted   : {format_hours(total_gpu_hours)}")
    lines.append(f"MACE raw wall-hours       : {format_hours(total_gpu_raw_wall_hours)}")
    lines.append(f"Ignored resume-gap hours  : {format_hours(total_ignored_gap_hours)}")
    lines.append("")
    lines.append("PARSE COUNTS")
    lines.append("-" * 80)
    lines.append(f"CRYSTAL .out files parsed : {n_crystal_ok}")
    lines.append(f"CRYSTAL .out failed       : {n_crystal_failed}")
    lines.append(f"train.log files parsed    : {n_gpu_ok}")
    lines.append(f"train.log failed          : {n_gpu_failed}")
    lines.append(f"CRYSTAL statuses          : {count_status(crystal_records)}")
    lines.append(f"train.log statuses        : {count_status(gpu_records)}")
    lines.append("")

    # Compact grouped report: by source/structure/functional/job_kind.
    compact: dict[tuple[str, str, str, str], dict[str, float | int | str]] = {}
    for row in grouped_rows:
        key = (
            str(row["source_type"]),
            str(row["structure"]),
            str(row["functional"]),
            str(row["job_kind"]),
        )
        if key not in compact:
            compact[key] = {
                "source_type": key[0],
                "structure": key[1],
                "functional": key[2],
                "job_kind": key[3],
                "n_files": 0,
                "n_ok": 0,
                "n_failed": 0,
                "cpu_hours": 0.0,
                "gpu_hours": 0.0,
            }
        compact[key]["n_files"] += int(row.get("n_files", 0) or 0)
        compact[key]["n_ok"] += int(row.get("n_ok", 0) or 0)
        compact[key]["n_failed"] += int(row.get("n_failed", 0) or 0)
        compact[key]["cpu_hours"] += float(row.get("cpu_hours", 0.0) or 0.0)
        compact[key]["gpu_hours"] += float(row.get("gpu_hours", 0.0) or 0.0)

    if compact:
        lines.append("GROUPED SUMMARY")
        lines.append("-" * 80)
        header = f"{'source':<14} {'structure':<14} {'functional':<8} {'kind':<9} {'files':>7} {'ok':>7} {'fail':>7} {'CPU h':>12} {'GPU h':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in sorted(compact.values(), key=lambda r: (str(r["source_type"]), str(r["structure"]), str(r["functional"]), str(r["job_kind"]))):
            lines.append(
                f"{str(row['source_type']):<14} "
                f"{str(row['structure']):<14} "
                f"{str(row['functional']):<8} "
                f"{str(row['job_kind']):<9} "
                f"{int(row['n_files']):>7d} "
                f"{int(row['n_ok']):>7d} "
                f"{int(row['n_failed']):>7d} "
                f"{float(row['cpu_hours']):>12.2f} "
                f"{float(row['gpu_hours']):>12.2f}"
            )
        lines.append("")

    failed_crystal = [rec for rec in crystal_records if rec.cpu_hours is None]
    failed_gpu = [rec for rec in gpu_records if rec.gpu_hours is None]
    if failed_crystal or failed_gpu:
        lines.append("FAILED / INCOMPLETE PARSES")
        lines.append("-" * 80)
        for rec in failed_crystal[:20]:
            lines.append(f"CRYSTAL failed: {rec.path} :: {rec.error}")
        if len(failed_crystal) > 20:
            lines.append(f"... {len(failed_crystal) - 20} more CRYSTAL failures omitted; see crystal_cpu_detail.csv")
        for rec in failed_gpu[:20]:
            lines.append(f"train.log failed: {rec.path} :: {rec.error}")
        if len(failed_gpu) > 20:
            lines.append(f"... {len(failed_gpu) - 20} more train.log failures omitted; see gpu_training_detail.csv")
        lines.append("")

    lines.append("Notes:")
    lines.append("- CRYSTAL CPU hours are core-hours from the final TOTAL CPU TIME line where available.")
    lines.append("- GPU hours use adjusted train.log wall-time times one GPU per run.")
    lines.append("- Large timestamp gaps above the threshold are treated as resume/idle gaps and excluded from adjusted GPU totals.")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize CRYSTAL CPU core-hours and MACE/MACELES GPU wall-hours."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root. Default: current directory.")
    parser.add_argument(
        "--structures-dir",
        action="append",
        default=None,
        help="Structure directory to scan for CRYSTAL .out files. Can be given multiple times. Default: structures and structures_pbe if present.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Results directory containing train.log files. Default: results.")
    parser.add_argument("--output-dir", type=Path, default=Path("time_summary"), help="Output directory. Default: time_summary.")
    parser.add_argument("--gap-hours", type=float, default=6.0, help="Timestamp gap threshold for resumed/appended train.log files. Default: 6 hours.")
    parser.add_argument("--gpus-per-run", type=int, default=1, help="GPU count per training run. Default: 1.")
    parser.add_argument("--include-slurm-out", action="store_true", help="Also scan slurm-*.out files under structure roots. Default: skip them.")
    return parser


def resolve_structure_roots(project_root: Path, structures_dir_args: Optional[list[str]]) -> list[Path]:
    if structures_dir_args:
        roots = [Path(p) for p in structures_dir_args]
    else:
        roots = [Path("structures"), Path("structures_pbe")]

    resolved: list[Path] = []
    for root in roots:
        if not root.is_absolute():
            root = project_root / root
        resolved.append(root)
    return resolved


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    project_root = args.project_root.resolve()
    structure_roots = resolve_structure_roots(project_root, args.structures_dir)
    results_root = args.results_dir if args.results_dir.is_absolute() else project_root / args.results_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    known_structures = list_known_structures(structure_roots)

    crystal_records = collect_crystal_cpu_records(
        structure_roots=structure_roots,
        known_structures=known_structures,
        include_slurm_out=args.include_slurm_out,
    )
    gpu_records = collect_gpu_training_records(
        results_root=results_root,
        known_structures=known_structures,
        gap_hours=args.gap_hours,
        gpus_per_run=args.gpus_per_run,
    )

    grouped_rows = group_crystal(crystal_records) + group_gpu(gpu_records)

    crystal_rows = [asdict(rec) for rec in crystal_records]
    gpu_rows = [asdict(rec) for rec in gpu_records]

    write_csv(output_dir / "crystal_cpu_detail.csv", crystal_rows)
    write_csv(output_dir / "gpu_training_detail.csv", gpu_rows)
    write_csv(output_dir / "time_summary_grouped.csv", grouped_rows)

    summary_payload = {
        "project_root": str(project_root),
        "structure_roots": [str(p) for p in structure_roots],
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "gap_hours": args.gap_hours,
        "gpus_per_run": args.gpus_per_run,
        "totals": {
            "crystal_cpu_hours": sum_optional(rec.cpu_hours for rec in crystal_records),
            "gpu_hours_adjusted": sum_optional(rec.gpu_hours for rec in gpu_records),
            "gpu_raw_wall_hours": sum_optional(rec.raw_wall_hours for rec in gpu_records),
            "ignored_gap_hours": sum_optional(rec.ignored_gap_hours for rec in gpu_records),
            "n_crystal_records": len(crystal_records),
            "n_gpu_records": len(gpu_records),
            "n_crystal_ok": sum(1 for rec in crystal_records if rec.cpu_hours is not None),
            "n_gpu_ok": sum(1 for rec in gpu_records if rec.gpu_hours is not None),
        },
        "crystal_status_counts": count_status(crystal_records),
        "gpu_status_counts": count_status(gpu_records),
        "grouped": grouped_rows,
    }
    (output_dir / "time_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    text_summary = make_text_summary(
        crystal_records=crystal_records,
        gpu_records=gpu_records,
        grouped_rows=grouped_rows,
        project_root=project_root,
        structure_roots=structure_roots,
        results_root=results_root,
        gap_hours=args.gap_hours,
        gpus_per_run=args.gpus_per_run,
    )
    (output_dir / "time_summary.txt").write_text(text_summary, encoding="utf-8")

    print(text_summary)
    print(f"Wrote outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
