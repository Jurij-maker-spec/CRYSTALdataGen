#!/usr/bin/env python3
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PYTHON_SCRIPTS_ROOT
sys.path.insert(0, str(PYTHON_SCRIPTS_ROOT))


from util.ref_db import (
    read_crystal_ir_reference,
    read_model_evaluation,
    list_model_evaluations_for_structure,
    format_hyperparams,
)
from util.plotting import restore_degeneracies, perform_KDE

DEFAULT_REF_DB = PROJECT_ROOT / "data" / "ref_db.h5"
DEFAULT_STYLE = PROJECT_ROOT / "util" / "style.mplstyle"
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "summary_plots"

STRUCTURE_PAIRS = [
    ("SiO2", "SiO2_PBE", r"SiO$_2$"),
    ("AlN", "AlN_PBE", "AlN"),
    ("Al2O3", "Al2O3_PBE", r"Al$_2$O$_3$"),
    ("TiO2_rutil", "TiO2_rutil_PBE", r"TiO$_2$ rutile"),
]

FUNCTIONAL = {
    'SiO2': r'SiO$_2$ HSEsol',
    'SiO2_PBE': r'SiO$2$ PBEsol',
    'TiO2_rutil': r'TiO$_2$ HSEsol',
    'TiO2_rutil_PBE': r'TiO$_2$ PBEsol',
    'Al2O3': r'Al$_2$O$_3$ HSEsol',
    'Al2O3_PBE': r'Al$_2$O$_3$ PBEsol',
    'AlN': 'AlN HSEsol',
    'AlN_PBE': 'AlN PBEsol'
}

CMAP = mpl.colormaps["managua"]


def _normalize_spectrum(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        return y

    ymax = np.nanmax(y)

    if not np.isfinite(ymax) or ymax <= 0.0:
        return y

    return y / ymax


def _parse_filter_value(raw: str):
    """
    Parse CLI hyperparameter filter values into int/float/string.
    """
    raw = str(raw)

    try:
        value = float(raw.replace("p", "."))
    except ValueError:
        return raw

    if value.is_integer():
        return int(value)

    return value


def _values_equal(a, b, *, atol: float = 1e-12) -> bool:
    """
    Compare hyperparameter values robustly.
    """
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()

    if isinstance(a, bytes):
        a = a.decode("utf-8")
    if isinstance(b, bytes):
        b = b.decode("utf-8")

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= atol

    return str(a) == str(b)


def _filter_rows_by_hyperparams(
    rows: list[dict],
    filters: dict[str, object] | None,
) -> list[dict]:
    """
    Keep only rows whose stored hyperparameters match all filters.
    """
    if not filters:
        return rows

    out = []

    for row in rows:
        hp = row.get("hyperparameters", {}) or {}
        keep = True

        for key, wanted in filters.items():
            if key not in hp:
                keep = False
                break

            if not _values_equal(hp[key], wanted):
                keep = False
                break

        if keep:
            out.append(row)

    return out


def _parse_hparam_filters(items: list[str] | None) -> dict[str, object]:
    """
    Parse CLI items like:
        --hparam rmax=4
        --hparam fw=75
        --hparam seed=2
    """
    if not items:
        return {}

    aliases = {
        "rmax": "r_max",
        "r_max": "r_max",
        "ew": "energy_weight",
        "energy_weight": "energy_weight",
        "fw": "forces_weight",
        "forces_weight": "forces_weight",
        "seed": "seed",
        "n": "train_size",
        "train_size": "train_size",
        "bs": "batch_size",
        "batch_size": "batch_size",
        "ep": "max_epochs",
        "max_epochs": "max_epochs",
    }

    filters = {}

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --hparam entry '{item}'. Expected key=value."
            )

        key, value = item.split("=", 1)
        key = aliases.get(key.strip(), key.strip())
        filters[key] = _parse_filter_value(value.strip())

    return filters


def print_hyperparameter_overview(rows: list[dict]) -> None:
    """
    Print available values for the most relevant hyperparameters.
    """
    keys = [
        "r_max",
        "energy_weight",
        "forces_weight",
        "seed",
        "train_size",
        "batch_size",
        "max_epochs",
    ]

    print("Available hyperparameters:")
    for key in keys:
        values = sorted({
            row.get("hyperparameters", {}).get(key)
            for row in rows
            if key in (row.get("hyperparameters", {}) or {})
        })

        if values:
            pretty = ", ".join(str(v) for v in values)
            print(f"  {key}: {pretty}")


def _select_rank_spaced(rows: list[dict], n: int) -> list[dict]:
    """
    Select n approximately equally spaced models from a ranked list.

    Always includes the best and worst available model.

    Example
    -------
    len(rows) = 81, n = 9
    selected ranks = 1, 11, 21, ..., 81
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    if not rows:
        return []

    if n >= len(rows):
        return rows

    indices = np.linspace(0, len(rows) - 1, n)
    indices = np.rint(indices).astype(int)

    # np.rint can theoretically create duplicates for small lists.
    # This removes duplicates while preserving order.
    unique_indices = []
    seen = set()

    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)

    return [rows[i] for i in unique_indices]


def apply_style(style_path: Path | None) -> None:
    if style_path is not None and style_path.exists():
        mpl.style.use(style_path)


def read_reference_spectrum(ref_db: Path, structure: str, fwhm: float):
    freqs, intensities = read_crystal_ir_reference(ref_db, structure)
    freqs, intensities, _ = restore_degeneracies(freqs, intensities)

    freqs = np.asarray(freqs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    mask = freqs > 1e-6
    freqs = freqs[mask]
    intensities = intensities[mask]

    if len(intensities) and np.max(intensities) > 0.0:
        intensities = intensities / np.max(intensities)

    kde, x = perform_KDE(freqs, intensities, FWHM=fwhm)

    if np.max(kde) > 0.0:
        kde = kde / np.max(kde)

    return x, kde


def _short_run_label(run_id: str, max_len: int = 44) -> str:
    """
    Keep long run names readable in stacked plots.
    """
    run_id = str(run_id)

    if len(run_id) <= max_len:
        return run_id

    return run_id[: max_len - 3] + "..."


def _position_text(ax, score, true_rank, fs=18, offset=1.1, base_x=0.01):
    ha_1 = 'left'
    ha_2 = 'right'

    ax.text(
        base_x,
        offset + 0.08,
        "rank:",
        transform=ax.get_yaxis_transform(),
        ha=ha_1,
        va="bottom",
        fontsize=fs,
    )
    ax.text(
        base_x + 0.071,
        offset + 0.08,
        f"{true_rank},",
        transform=ax.get_yaxis_transform(),
        ha=ha_2,
        va="bottom",
        fontsize=fs,
    )
    ax.text(
        base_x + 0.085,
        offset + 0.08,
        "score:",
        transform=ax.get_yaxis_transform(),
        ha=ha_1,
        va="bottom",
        fontsize=fs,
    )
    ax.text(
        base_x + 0.171,
        offset + 0.08,
        f"{score:.1f}",
        transform=ax.get_yaxis_transform(),
        ha=ha_2,
        va="bottom",
        fontsize=fs,
    )




def plot_hse_pbe_ir_grid(
    *,
    ref_db: Path,
    outdir: Path,
    outfile_stem: str,
    fwhm: float,
    style_path: Path | None,
) -> None:
    apply_style(style_path)

    colors = CMAP([0.1, 0.8])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        sharey=True,
        constrained_layout=True,
    )
    i = 0
    for ax, (hse_name, pbe_name, title) in zip(axes.ravel(), STRUCTURE_PAIRS):
        x_hse, y_hse = read_reference_spectrum(ref_db, hse_name, fwhm)
        x_pbe, y_pbe = read_reference_spectrum(ref_db, pbe_name, fwhm)

        ax.plot(x_hse, y_hse, lw=2, color=colors[1], label="HSEsol" if i == 0 else '')
        ax.plot(x_pbe, y_pbe, lw=2, ls='--', color=colors[0], label="PBEsol" if i == 0 else '')
        ax.text(
            0.50,
            0.95,
            title,
            transform=ax.transAxes,
            va='top', 
            ha='center',
            size=22,
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.85), 
            )
        i += 1
    
    fig.supylabel('Relative IR intensity')
    fig.supxlabel(r"Wavenumber in cm$^{-1}$")
    fig.legend(bbox_to_anchor=(0.65, 0.65), fontsize=24, framealpha=0.9, shadow=True)
    # fig.suptitle('Comparison of reference  methods')

    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / f"{outfile_stem}.png"
    pdf = outdir / f"{outfile_stem}.pdf"

    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_top_N_of_a_structure_old(
    *,
    ref_db: Path,
    structure: str,
    outdir: Path,
    outfile_stem: str | None = None,
    top_n: int = 9,
    metric: str = "composite_score",
    selection: str = "top",
    fwhm: float = 12.0,
    style_path: Path | None = None,
    y_offset: float = 1.15,
    normalize_models: bool = True,
    reference_at_bottom: bool = True,
) -> None:
    """
    Plot the top N cached model IR spectra for one structure.

    The selected runs are ranked by `metric`. For lower-is-better metrics like
    composite_score, the best N are selected. They are then plotted from
    worst-of-top-N at the top to best at the bottom, directly above the
    CRYSTAL reference.

    Layout:
        worst selected model
        ...
        best selected model
        CRYSTAL reference
    """
    apply_style(style_path)
    fs = 16
    ref_db = Path(ref_db)
    outdir = Path(outdir)

    if "TiO2" in structure:
        x_base = 0.8
        tio2_lim = True
    else:
        x_base = 0.01
        tio2_lim = False

    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    rows = list_model_evaluations_for_structure(
        ref_db,
        structure,
        metric=metric,
        require_ir_spectrum=True,
        require_metric=True,
        sort=True,
    )

    if not rows:
        raise ValueError(
            f"No model evaluations with metric '{metric}' and IR spectra "
            f"found for structure '{structure}'."
        )

    if selection == "top":
        selected = rows[:top_n]

    elif selection == "spaced":
        selected = _select_rank_spaced(rows, top_n)

    else:
        raise ValueError(
            f"Unknown selection mode: {selection}. "
            "Use 'top' or 'spaced'."
        )

    # Plot worst selected model at top and best selected model closest to CRYSTAL.
    # rows are ranked best -> worst, so selected is also ordered best -> worst.
    selected_for_plot = list(reversed(selected))

    x_ref, y_ref = read_reference_spectrum(ref_db, structure, fwhm)
    y_ref = _normalize_spectrum(y_ref)

    n_models = len(selected_for_plot)

    fig_height = max(4.5, 0.55 * (n_models + 1))
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(14, fig_height),
        constrained_layout=True,
    )

    colors = CMAP(np.linspace(0.15, 0.85, max(n_models, 2)))

    # ------------------------------------------------------------
    # CRYSTAL reference at bottom
    # ------------------------------------------------------------
    ref_offset = 0.0
    ax.plot(
        x_ref,
        y_ref + ref_offset,
        color="black",
        lw=1.2,
        label="CRYSTAL reference",
        zorder=5,
    )
    

    ax.text(
        x_base,
        ref_offset + 0.08,
        f"CRYSTAL {FUNCTIONAL[structure]}",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=fs,
    )

    # ------------------------------------------------------------
    # Model spectra
    # ------------------------------------------------------------
    ytick_positions = [ref_offset]
    ytick_labels = ["0"]

    for plot_idx, row in enumerate(selected_for_plot):
        # plot_idx = 0 is worst-of-selected.
        # Put it highest; best ends up closest to CRYSTAL.
        offset = (n_models - plot_idx) * y_offset

        ev = read_model_evaluation(
            ref_db,
            structure,
            run_id=row["run_id"],
            dataset_split=row["dataset_split"],
            sweep_id=row["sweep_id"],
        )

        x = np.asarray(ev["nu_grid_cm1"], dtype=float)
        y = np.asarray(ev["ir_spec"], dtype=float)

        if normalize_models:
            y = _normalize_spectrum(y)

        # Rank in true sorted order: 1 = best.
        true_rank = rows.index(row) + 1
        metric_value = row["metric_value"]

        label = (
            f"{true_rank:4d}: {metric}={metric_value:.3g}, "
            #f"{_short_run_label(row['run_id'])}"
        )

        ax.plot(
            x,
            y + offset,
            lw=1.5,
            color=colors[plot_idx],
            label=label,
        )

        _position_text(ax, metric_value, true_rank, offset=offset, base_x=x_base)

        ytick_positions.append(offset)
        ytick_labels.append(f"{true_rank}")


    if tio2_lim:
        ax.set_xlim(0,820)
    # ------------------------------------------------------------
    # Axes and labels
    # ------------------------------------------------------------
    ax.set_xlabel(r"Wavenumber in cm$^{-1}$")
    ax.set_ylabel("Relative IR intensity")

    title_metric = metric.replace("_", r"\_")
    # ax.set_title(rf"{structure}: top {len(selected)} model IR spectra ranked by {title_metric}")

    ax.set_yticks(ytick_positions)
    # ax.set_yticklabels(ytick_labels)
    ax.set_yticklabels('')

    ax.set_ylim(-0.10, (n_models + 1) * y_offset)
    ax.margins(x=0.01)

    # Keep legend outside because run IDs are long.
    # ax.legend(
    #     loc="center left",
    #     bbox_to_anchor=(1.01, 0.5),
    #     fontsize=8,
    #     frameon=False,
    # )

    outdir.mkdir(parents=True, exist_ok=True)

    if outfile_stem is None:
        safe_metric = metric.replace("/", "_")
        outfile_stem = f"{structure}_{selection}_{len(selected)}_{safe_metric}_ir_spectra"

    png = outdir / f"{outfile_stem}.png"
    pdf = outdir / f"{outfile_stem}.pdf"

    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    print()
    print(f"Top {len(selected)} runs for {structure} by {metric}:")
    for i, row in enumerate(selected, start=1):
        print(
            f"{i:>2d}. {row['metric_value']:.6g}  "
            f"{row['dataset_split']} / {row['sweep_id']} / {row['run_id']}"
        )


def plot_top_N_of_a_structure(
    *,
    ref_db: Path,
    structure: str,
    outdir: Path,
    outfile_stem: str | None = None,
    top_n: int = 9,
    metric: str = "composite_score",
    selection: str = "top",
    hparam_filters: dict[str, object] | None = None,
    print_hparams: bool = False,
    fwhm: float = 12.0,
    style_path: Path | None = None,
    y_offset: float = 1.15,
    normalize_models: bool = True,
    reference_at_bottom: bool = True,
) -> None:
    """
    Plot selected cached model IR spectra for one structure.

    The model list is first ranked by `metric`.

    Optional hyperparameter filtering is applied before selection, e.g.
        hparam_filters={"r_max": 4.0}

    Selection modes:
        selection="top"
            Select the best N models after filtering.

        selection="spaced"
            Select N approximately equally spaced models from the filtered ranking.

    Plot layout:
        worst selected model
        ...
        best selected model
        CRYSTAL reference
    """
    apply_style(style_path)
    fs = 16

    ref_db = Path(ref_db)
    outdir = Path(outdir)

    if "TiO2" in structure:
        x_base = 0.8
        tio2_lim = True
    else:
        x_base = 0.01
        tio2_lim = False

    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    # ------------------------------------------------------------
    # Load all ranked rows
    # ------------------------------------------------------------
    rows_all = list_model_evaluations_for_structure(
        ref_db,
        structure,
        metric=metric,
        require_ir_spectrum=True,
        require_metric=True,
        sort=True,
    )

    if not rows_all:
        raise ValueError(
            f"No model evaluations with metric '{metric}' and IR spectra "
            f"found for structure '{structure}'."
        )

    if print_hparams:
        print_hyperparameter_overview(rows_all)
        print()

    # ------------------------------------------------------------
    # Hyperparameter filtering happens before top/spaced selection
    # ------------------------------------------------------------
    rows = _filter_rows_by_hyperparams(rows_all, hparam_filters)

    if not rows:
        raise ValueError(
            f"No model evaluations left after filtering for structure "
            f"'{structure}' with filters {hparam_filters}."
        )

    # ------------------------------------------------------------
    # Select models
    # ------------------------------------------------------------
    if selection == "top":
        selected = rows[:top_n]

    elif selection == "spaced":
        selected = _select_rank_spaced(rows, top_n)

    else:
        raise ValueError(
            f"Unknown selection mode: {selection}. "
            "Use 'top' or 'spaced'."
        )

    # Plot worst selected model at top and best selected model closest to CRYSTAL.
    # rows are ranked best -> worst, so selected is also ordered best -> worst.
    selected_for_plot = list(reversed(selected))

    # ------------------------------------------------------------
    # Reference spectrum
    # ------------------------------------------------------------
    x_ref, y_ref = read_reference_spectrum(ref_db, structure, fwhm)
    y_ref = _normalize_spectrum(y_ref)

    n_models = len(selected_for_plot)

    fig_height = max(4.5, 0.55 * (n_models + 1))
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(14, fig_height),
        constrained_layout=True,
    )

    colors = CMAP(np.linspace(0.15, 0.85, max(n_models, 2)))

    # ------------------------------------------------------------
    # CRYSTAL reference at bottom
    # ------------------------------------------------------------
    ref_offset = 0.0
    ax.plot(
        x_ref,
        y_ref + ref_offset,
        color="black",
        lw=1.2,
        label="CRYSTAL reference",
        zorder=5,
    )

    ax.text(
        x_base,
        ref_offset + 0.08,
        f"CRYSTAL {FUNCTIONAL[structure]}",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=fs,
    )

    # ------------------------------------------------------------
    # Model spectra
    # ------------------------------------------------------------
    ytick_positions = [ref_offset]
    ytick_labels = ["0"]

    for plot_idx, row in enumerate(selected_for_plot):
        # plot_idx = 0 is worst-of-selected.
        # Put it highest; best ends up closest to CRYSTAL.
        offset = (n_models - plot_idx) * y_offset

        ev = read_model_evaluation(
            ref_db,
            structure,
            run_id=row["run_id"],
            dataset_split=row["dataset_split"],
            sweep_id=row["sweep_id"],
        )

        x = np.asarray(ev["nu_grid_cm1"], dtype=float)
        y = np.asarray(ev["ir_spec"], dtype=float)

        if normalize_models:
            y = _normalize_spectrum(y)

        global_rank = rows_all.index(row) + 1
        filtered_rank = rows.index(row) + 1
        metric_value = row["metric_value"]

        hp_label = row.get("hyperparam_label", "")
        if not hp_label:
            hp_label = format_hyperparams(row.get("hyperparameters", {}))

        label = f"{global_rank:4d}: {metric}={metric_value:.3g}"

        ax.plot(
            x,
            y + offset,
            lw=1.5,
            color=colors[plot_idx],
            label=label,
        )

        # Preferred if you update _position_text to accept extra_text.
        _position_text(
            ax,
            metric_value,
            global_rank,
            offset=offset,
            base_x=x_base,
            #extra_text=hp_label,
            #filtered_rank=filtered_rank,
        )

        ytick_positions.append(offset)
        ytick_labels.append(f"{global_rank}")

    if tio2_lim:
        ax.set_xlim(0, 820)

    # ------------------------------------------------------------
    # Axes and labels
    # ------------------------------------------------------------
    ax.set_xlabel(r"Wavenumber in cm$^{-1}$")
    ax.set_ylabel("Relative IR intensity")

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels("")

    ax.set_ylim(-0.10, (n_models + 1) * y_offset)
    ax.margins(x=0.01)

    outdir.mkdir(parents=True, exist_ok=True)

    if outfile_stem is None:
        safe_metric = metric.replace("/", "_")

        filter_suffix = ""
        if hparam_filters:
            filter_suffix = "_" + "_".join(
                f"{k}-{v}" for k, v in hparam_filters.items()
            )
            filter_suffix = filter_suffix.replace("/", "_")

        outfile_stem = (
            f"{structure}_{selection}_{len(selected)}_"
            f"{safe_metric}{filter_suffix}_ir_spectra"
        )

    png = outdir / f"{outfile_stem}.png"
    pdf = outdir / f"{outfile_stem}.pdf"

    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    print()
    print(f"Selected {len(selected)} runs for {structure} by {metric}:")
    if hparam_filters:
        print(f"Filters: {hparam_filters}")

    for row in selected:
        global_rank = rows_all.index(row) + 1
        filtered_rank = rows.index(row) + 1

        hp_label = row.get("hyperparam_label", "")
        if not hp_label:
            hp_label = format_hyperparams(row.get("hyperparameters", {}))

        print(
            f"global rank {global_rank:>4d} | "
            f"filtered rank {filtered_rank:>4d} | "
            f"{metric} {row['metric_value']:>10.5g} | "
            f"{hp_label} | "
            f"{row['dataset_split']} / {row['sweep_id']} / {row['run_id']}"
        )



def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ref-db", type=Path, default=DEFAULT_REF_DB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project-level plotting CLI for CRYSTALdataGen."
    )

    sub = parser.add_subparsers(dest="command", required=True)
    p_grid = sub.add_parser(
        "ir-grid",
        help="Plot HSEsol vs PBEsol CRYSTAL reference IR spectra in a 2x2 grid.",
    )
    add_common_args(p_grid)
    p_grid.add_argument("--outfile-stem", default="hse_pbe_reference_ir_grid")
    p_grid.add_argument("--fwhm", type=float, default=12.0)

    p_top_ir = sub.add_parser(
        "top-n-ir",
        help="Plot selected cached model IR spectra for one structure.",
    )
    add_common_args(p_top_ir)
    p_top_ir.add_argument("--structure", required=True)
    p_top_ir.add_argument("--top-n", type=int, default=9)
    p_top_ir.add_argument("--metric", default="composite_score")
    p_top_ir.add_argument(
        "--selection",
        choices=["top", "spaced"],
        default="top",
        help=(
            "'top' selects the best N models after filtering. "
            "'spaced' selects N equally spaced models after filtering."
        ),
    )
    p_top_ir.add_argument(
        "--hparam",
        action="append",
        default=None,
        help=(
            "Filter by hyperparameter. Can be given multiple times. "
            "Examples: --hparam rmax=4 --hparam fw=75 --hparam seed=2"
        ),
    )
    p_top_ir.add_argument(
        "--print-hparams",
        action="store_true",
        help="Print available hyperparameter values before plotting.",
    )
    p_top_ir.add_argument("--outfile-stem", default=None)
    p_top_ir.add_argument("--fwhm", type=float, default=12.0)
    p_top_ir.add_argument("--y-offset", type=float, default=1.15)
    p_top_ir.add_argument(
        "--no-normalize-models",
        action="store_true",
        help="Use stored model spectrum amplitudes without renormalizing each spectrum.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "ir-grid":
        plot_hse_pbe_ir_grid(
            ref_db=args.ref_db,
            outdir=args.outdir,
            outfile_stem=args.outfile_stem,
            fwhm=args.fwhm,
            style_path=args.style,
        )

    elif args.command == "top-n-ir":
        hparam_filters = _parse_hparam_filters(args.hparam)

        plot_top_N_of_a_structure(
            ref_db=args.ref_db,
            structure=args.structure,
            outdir=args.outdir,
            outfile_stem=args.outfile_stem,
            top_n=args.top_n,
            metric=args.metric,
            selection=args.selection,
            hparam_filters=hparam_filters,
            print_hparams=args.print_hparams,
            fwhm=args.fwhm,
            style_path=args.style,
            y_offset=args.y_offset,
            normalize_models=not args.no_normalize_models,
        )

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

