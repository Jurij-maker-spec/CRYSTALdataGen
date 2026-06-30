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
    read_crystal_modes,
    read_model_evaluation,
    read_mode_overlap,
    read_mode_matches,
    read_ir_peak_matches,
    list_model_evaluations_for_structure,
    format_hyperparams,
)
from util.plotting import (
    restore_degeneracies,
    plot_ir_spectrum_with_frequency_correlation,
    perform_KDE,
)
from util.mode_matching import (
    compare_mode_sets,
    plot_combined_overlap_heatmaps,
    _slice_CMAP,
)

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
    'SiO2_PBE': r'SiO$_2$ PBEsol',
    'TiO2_rutil': r'TiO$_2$ HSEsol',
    'TiO2_rutil_PBE': r'TiO$_2$ PBEsol',
    'Al2O3': r'Al$_2$O$_3$ HSEsol',
    'Al2O3_PBE': r'Al$_2$O$_3$ PBEsol',
    'AlN': 'AlN HSEsol',
    'AlN_PBE': 'AlN PBEsol'
}

CMAP = mpl.colormaps["managua"]

###########################################################################
# HELPERS
###########################################################################


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
        base_x + 0.082,
        offset + 0.08,
        f"{true_rank},",
        transform=ax.get_yaxis_transform(),
        ha=ha_2,
        va="bottom",
        fontsize=fs,
    )
    ax.text(
        base_x + 0.097,
        offset + 0.08,
        "score:",
        transform=ax.get_yaxis_transform(),
        ha=ha_1,
        va="bottom",
        fontsize=fs,
    )
    ax.text(
        base_x + 0.192,
        offset + 0.08,
        f"{score:.1f}",
        transform=ax.get_yaxis_transform(),
        ha=ha_2,
        va="bottom",
        fontsize=fs,
    )


def _reference_ir_intensity_at_mode_frequency(
    *,
    ref_db: Path,
    structure: str,
    ref_freqs_cm1: np.ndarray,
    tolerance_cm1: float = 2.0,
) -> np.ndarray:
    """
    Map full reference-mode frequencies to normalized CRYSTAL IR intensities.

    For all-mode plots, many modes are IR-inactive. Those receive intensity 0.
    Matching is frequency-based because the stored CRYSTAL IR reference is
    stored as IR frequencies + intensities, not necessarily as full 3N mode
    intensity array.
    """
    ref_freqs_cm1 = np.asarray(ref_freqs_cm1, dtype=float)

    out = np.zeros_like(ref_freqs_cm1, dtype=float)

    try:
        ir_freqs, ir_intensities = read_crystal_ir_reference(ref_db, structure)
        ir_freqs, ir_intensities, _ = restore_degeneracies(ir_freqs, ir_intensities)

        ir_freqs = np.asarray(ir_freqs, dtype=float)
        ir_intensities = np.asarray(ir_intensities, dtype=float)

        mask = ir_freqs > 1e-6
        ir_freqs = ir_freqs[mask]
        ir_intensities = ir_intensities[mask]

        if ir_freqs.size == 0:
            return out

        imax = np.nanmax(ir_intensities)
        if np.isfinite(imax) and imax > 0.0:
            ir_intensities = ir_intensities / imax

        for i, f in enumerate(ref_freqs_cm1):
            j = int(np.argmin(np.abs(ir_freqs - f)))
            if abs(ir_freqs[j] - f) <= tolerance_cm1:
                out[i] = ir_intensities[j]

    except Exception:
        pass

    return out


def _marker_sizes_from_intensity(
    intensity_rel: np.ndarray,
    *,
    min_size: float = 10.0,
    max_size: float = 140.0,
) -> np.ndarray:
    intensity_rel = np.asarray(intensity_rel, dtype=float)
    intensity_rel = np.nan_to_num(intensity_rel, nan=0.0, posinf=0.0, neginf=0.0)
    intensity_rel = np.clip(intensity_rel, 0.0, None)

    imax = np.nanmax(intensity_rel) if intensity_rel.size else 0.0
    if imax > 0.0:
        intensity_rel = intensity_rel / imax

    # sqrt scaling keeps weak IR modes visible without making strong modes huge.
    return min_size + (max_size - min_size) * np.sqrt(intensity_rel)


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    vmax = np.nanmax(values)
    if not np.isfinite(vmax) or vmax <= 0.0:
        return values

    return values / vmax


def _collect_frequency_evolution_points(
    *,
    ref_db: Path,
    structure: str,
    rows_all: list[dict],
    selected_rows: list[dict],
    rows_filtered: list[dict],
    metric: str,
    modes: str,
    y_axis: str,
    color_by: str,
    skip_first: int,
    ir_intensity_tolerance_cm1: float,
) -> dict[str, np.ndarray]:
    """
    Convert selected model rows into scatter-plot arrays.
    """
    xs = []
    ys = []
    cs = []
    ss_intensity = []
    ref_freqs_all = []
    ref_mode_indices_all = []

    for row in selected_rows:
        global_rank = rows_all.index(row) + 1
        filtered_rank = rows_filtered.index(row) + 1
        n_total = len(rows_all)

        if y_axis == "rank":
            y_value = float(global_rank)
        elif y_axis == "filtered_rank":
            y_value = float(filtered_rank)
        elif y_axis == "rank_percentile":
            if n_total <= 1:
                y_value = 0.0
            else:
                y_value = 100.0 * (global_rank - 1) / (n_total - 1)
        elif y_axis == "metric":
            y_value = float(row["metric_value"])
        else:
            raise ValueError(
                f"Unknown y_axis={y_axis}. "
                "Use rank, filtered_rank, rank_percentile, or metric."
            )

        if modes == "matched_modes":
            matches = read_mode_matches(
                ref_db,
                structure=structure,
                run_id=row["run_id"],
                dataset_split=row["dataset_split"],
                sweep_id=row["sweep_id"],
            )

            ref_idx = np.asarray(matches["ref_mode_index"], dtype=int)
            ref_freq = np.asarray(matches["ref_freq_cm1"], dtype=float)
            model_freq = np.asarray(matches["model_freq_cm1"], dtype=float)
            abs_delta = np.asarray(matches["abs_delta_cm1"], dtype=float)
            overlap = np.asarray(matches["overlap"], dtype=float)

            mask = (
                (ref_idx >= skip_first)
                & np.isfinite(ref_freq)
                & np.isfinite(model_freq)
                & (ref_freq > 1e-6)
                & (model_freq > 1e-6)
            )

            ref_idx = ref_idx[mask]
            ref_freq = ref_freq[mask]
            model_freq = model_freq[mask]
            abs_delta = abs_delta[mask]
            overlap = overlap[mask]

            ref_intensity = _reference_ir_intensity_at_mode_frequency(
                ref_db=ref_db,
                structure=structure,
                ref_freqs_cm1=ref_freq,
                tolerance_cm1=ir_intensity_tolerance_cm1,
            )

            if color_by == "abs_delta_cm1":
                color_values = abs_delta
            elif color_by == "one_minus_overlap":
                color_values = 1.0 - overlap
            elif color_by == "overlap":
                color_values = overlap
            else:
                raise ValueError(
                    f"color_by={color_by} is not available for modes='matched_modes'. "
                    "Use abs_delta_cm1, one_minus_overlap, or overlap."
                )

        elif modes == "ir_active":
            matches = read_ir_peak_matches(
                ref_db,
                structure=structure,
                run_id=row["run_id"],
                dataset_split=row["dataset_split"],
                sweep_id=row["sweep_id"],
            )

            ref_freq = np.asarray(matches["ref_freq_cm1"], dtype=float)
            model_freq = np.asarray(matches["model_freq_cm1"], dtype=float)
            abs_delta = np.asarray(matches["abs_delta_cm1"], dtype=float)

            mask = (
                np.isfinite(ref_freq)
                & np.isfinite(model_freq)
                & (ref_freq > 1e-6)
                & (model_freq > 1e-6)
            )

            ref_freq = ref_freq[mask]
            model_freq = model_freq[mask]
            abs_delta = abs_delta[mask]

            if "ref_intensity" in matches:
                ref_intensity = np.asarray(matches["ref_intensity"], dtype=float)[mask]
                ref_intensity = _normalize_positive(ref_intensity)
            else:
                ref_intensity = np.ones_like(ref_freq)

            if color_by == "abs_delta_cm1":
                color_values = abs_delta

            elif color_by == "intensity_abs_error":
                if "ref_intensity" not in matches or "model_intensity" not in matches:
                    raise KeyError(
                        "color_by='intensity_abs_error' requires ref_intensity "
                        "and model_intensity in ir_matching/matched_peaks."
                    )

                ref_i = np.asarray(matches["ref_intensity"], dtype=float)[mask]
                mod_i = np.asarray(matches["model_intensity"], dtype=float)[mask]

                ref_i = _normalize_positive(ref_i)
                mod_i = _normalize_positive(mod_i)

                color_values = np.abs(mod_i - ref_i)

            else:
                raise ValueError(
                    f"color_by={color_by} is not available for modes='ir_active'. "
                    "Use abs_delta_cm1 or intensity_abs_error."
                )

            # No true full-mode index exists in the IR matching table.
            ref_idx = np.arange(len(ref_freq), dtype=int)

        else:
            raise ValueError("modes must be 'matched_modes' or 'ir_active'.")

        n = len(model_freq)

        xs.append(model_freq)
        ys.append(np.full(n, y_value, dtype=float))
        cs.append(color_values)
        ss_intensity.append(ref_intensity)
        ref_freqs_all.append(ref_freq)
        ref_mode_indices_all.append(ref_idx)

    if not xs:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "c": np.array([]),
            "intensity": np.array([]),
            "ref_freq": np.array([]),
            "ref_mode_index": np.array([]),
        }

    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "c": np.concatenate(cs),
        "intensity": np.concatenate(ss_intensity),
        "ref_freq": np.concatenate(ref_freqs_all),
        "ref_mode_index": np.concatenate(ref_mode_indices_all),
    }


def _reference_lines_for_frequency_evolution(
    *,
    ref_db: Path,
    structure: str,
    modes: str,
    skip_first: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return reference-line frequencies and normalized reference IR intensities.

    The second array can be used to vary alpha/linewidth later.
    """
    if modes == "ir_active":
        freqs, intensities = read_crystal_ir_reference(ref_db, structure)
        freqs, intensities, _ = restore_degeneracies(freqs, intensities)

        freqs = np.asarray(freqs, dtype=float)
        intensities = np.asarray(intensities, dtype=float)

        mask = freqs > 1e-6
        freqs = freqs[mask]
        intensities = intensities[mask]

        intensities = _normalize_positive(intensities)
        return freqs, intensities

    if modes == "matched_modes":
        modes_ref = read_crystal_modes(ref_db, structure)
        freqs = np.asarray(modes_ref["freqs_cm"], dtype=float)

        if skip_first > 0:
            freqs = freqs[skip_first:]

        freqs = freqs[freqs > 1e-6]

        intensities = _reference_ir_intensity_at_mode_frequency(
            ref_db=ref_db,
            structure=structure,
            ref_freqs_cm1=freqs,
        )

        return freqs, intensities

    raise ValueError("modes must be 'matched_modes' or 'ir_active'.")


def _draw_reference_frequency_lines(
    ax,
    *,
    ref_freqs: np.ndarray,
    ref_intensities: np.ndarray,
    color: str = "black",
) -> None:
    ref_freqs = np.asarray(ref_freqs, dtype=float)
    ref_intensities = np.asarray(ref_intensities, dtype=float)

    for f, inten in zip(ref_freqs, ref_intensities):
        # IR-active modes get slightly stronger guide lines.
        alpha = 0.12 + 0.28 * float(np.clip(inten, 0.0, 1.0))
        lw = 0.5 + 0.8 * float(np.clip(inten, 0.0, 1.0))

        ax.axvline(
            f,
            ls="--",
            lw=lw,
            color=color,
            alpha=alpha,
            zorder=0,
        )


def _draw_frequency_error_bands(
    ax,
    *,
    points: dict[str, np.ndarray],
    max_width_cm1: float = 60.0,
    min_width_cm1: float = 0.0,
) -> None:
    """
    Draw a light band around each reference frequency.

    Band half-width = median |model_freq - ref_freq| across selected models,
    grouped by the rounded reference frequency.
    """
    ref_freq = np.asarray(points["ref_freq"], dtype=float)
    x = np.asarray(points["x"], dtype=float)

    if ref_freq.size == 0:
        return

    rounded = np.round(ref_freq, decimals=4)

    for rf in np.unique(rounded):
        mask = rounded == rf
        if not np.any(mask):
            continue

        deltas = np.abs(x[mask] - ref_freq[mask])
        width = float(np.nanmedian(deltas))

        if not np.isfinite(width):
            continue

        width = np.clip(width, min_width_cm1, max_width_cm1)

        if width <= 0.0:
            continue

        ax.axvspan(
            float(rf) - width,
            float(rf) + width,
            alpha=0.1,
            color="black",
            lw=0.0,
            zorder=0,
        )


def _resolve_structure_pair(structure_base: str) -> tuple[str, str, str]:
    """
    Resolve HSEsol/PBEsol pair from STRUCTURE_PAIRS.
    """
    for hse_name, pbe_name, label in STRUCTURE_PAIRS:
        if structure_base in {hse_name, pbe_name}:
            return hse_name, pbe_name, label

    if structure_base.endswith("_PBE"):
        return structure_base[:-4], structure_base, structure_base[:-4]

    return structure_base, f"{structure_base}_PBE", structure_base



###########################################################################
# PLOTTERS
###########################################################################


def plot_hse_pbe_ir_grid(
    *,
    ref_db: Path,
    outdir: Path,
    outfile_stem: str,
    fwhm: float,
    style_path: Path | None,
) -> None:
    apply_style(style_path)

    colors = CMAP([0.2, 0.65])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 6),
        sharey=True,
        constrained_layout=True,
    )
    i = 0
    abcdlabel = ['a)', 'b)', 'c)', 'd)']
    for ax, (hse_name, pbe_name, title) in zip(axes.ravel(), STRUCTURE_PAIRS):
        x_hse, y_hse = read_reference_spectrum(ref_db, hse_name, fwhm)
        x_pbe, y_pbe = read_reference_spectrum(ref_db, pbe_name, fwhm)

        ax.plot(x_hse, y_hse, lw=2, color=colors[1], label="HSEsol" if i == 0 else '')
        ax.fill_between(x_hse, y_hse, color=colors[1], alpha=0.1)
        ax.plot(x_pbe, y_pbe, lw=2, ls='--', color=colors[0], label="PBEsol" if i == 0 else '')
        ax.fill_between(x_pbe, y_pbe, color=colors[0], alpha=0.1)
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
        ax.text(
            0.05,
            0.95,
            abcdlabel[i],
            transform=ax.transAxes,
            va='top', 
            ha='center',
            size=24,
        )
        i += 1
    
    fig.supylabel('Relative IR intensity')
    fig.supxlabel(r"Wavenumber in cm$^{-1}$")
    # fig.legend(bbox_to_anchor=(0.65, 0.65), fontsize=24, framealpha=0.9, shadow=True)
    fig.legend(
        loc="lower center", 
        ncol=2, 
        bbox_to_anchor=(0.5, 0.98), 
        fontsize=22, 
        framealpha=0.9, 
        frameon=True,
    )
    # fig.suptitle('Comparison of reference  methods')

    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / f"{outfile_stem}.png"
    pdf = outdir / f"{outfile_stem}.pdf"

    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


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
    scale: float = 0.85
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
        figsize=(14*scale, fig_height*scale),
        constrained_layout=True,
    )

    colors = CMAP(np.linspace(0.15, 0.85, max(n_models, 2)))

    # ------------------------------------------------------------
    # CRYSTAL reference at bottom
    # ------------------------------------------------------------
    fill_offset=0.025
    ref_offset = 0.0
    ax.plot(
        x_ref,
        y_ref + ref_offset,
        ref_offset-fill_offset,
        color="black",
        lw=1.2,
        label="CRYSTAL reference",
        zorder=5,
    )
    ax.fill_between(
        x_ref, 
        y_ref+ref_offset,
        ref_offset,
        color = "black",
        alpha = 0.1
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
        ax.fill_between(
            x,
            y+offset,
            offset-fill_offset,
            color=colors[plot_idx],
            alpha=0.2
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


def plot_best_model_result_figures_for_structure(
    *,
    ref_db: Path,
    structure: str,
    outdir_root: Path = Path("/home/jha/jha/python_scripts/master_thesis/figures/results"),
    metric: str = "composite_score",
    include_functional_pair: bool = True,
    hparam_filters: dict[str, object] | None = None,
    style_path: Path | None = DEFAULT_STYLE,
    normalize_ir: bool = True,
    skip_first: int = 3,
    degeneracy_tol: float = 0.5,  # kept for CLI compatibility; not used with cached overlap
    skip_missing: bool = True,
) -> dict[str, dict]:
    """
    Plot thesis/result figures for the best cached model of a structure.

    For each selected structure, this writes:

        <structure>_ir_corr_comp.pdf
        <structure>_mode_overlap_combined.pdf

    The mode-overlap plot uses cached mode_matching data from ref_db.
    """
    apply_style(style_path)

    ref_db = Path(ref_db)
    outdir_root = Path(outdir_root)

    # ------------------------------------------------------------
    # Resolve HSEsol/PBEsol pair.
    # ------------------------------------------------------------
    base_structure = structure
    structures_to_plot = [structure]

    for hse_name, pbe_name, _label in STRUCTURE_PAIRS:
        if structure in {hse_name, pbe_name}:
            base_structure = hse_name
            structures_to_plot = [hse_name, pbe_name] if include_functional_pair else [structure]
            break
    else:
        if structure.endswith("_PBE"):
            base_structure = structure[:-4]

    outdir = outdir_root / base_structure
    outdir.mkdir(parents=True, exist_ok=True)

    written: dict[str, dict] = {}

    for struct in structures_to_plot:
        # --------------------------------------------------------
        # Select best cached model.
        # --------------------------------------------------------
        rows_all = list_model_evaluations_for_structure(
            ref_db,
            struct,
            metric=metric,
            require_ir_spectrum=True,
            require_metric=True,
            sort=True,
        )

        if not rows_all:
            msg = (
                f"No cached model evaluations with metric '{metric}' and "
                f"IR spectrum found for structure '{struct}'."
            )
            if skip_missing:
                print(f"WARNING: {msg}")
                continue
            raise ValueError(msg)

        rows = _filter_rows_by_hyperparams(rows_all, hparam_filters)

        if not rows:
            msg = (
                f"No cached model evaluations left after filtering for "
                f"structure '{struct}' with filters {hparam_filters}."
            )
            if skip_missing:
                print(f"WARNING: {msg}")
                continue
            raise ValueError(msg)

        best = rows[0]
        best_rank = rows_all.index(best) + 1

        # --------------------------------------------------------
        # Read exact cached model data from ref_db.
        # Keys are defined by util.ref_db.read_model_evaluation().
        # --------------------------------------------------------
        ev = read_model_evaluation(
            ref_db,
            struct,
            run_id=best["run_id"],
            dataset_split=best["dataset_split"],
            sweep_id=best["sweep_id"],
        )

        freqs_model = np.asarray(ev["frequencies_cm1"], dtype=float)
        intensities = np.asarray(ev["intensities"], dtype=float)
        nu_grid = np.asarray(ev["nu_grid_cm1"], dtype=float)
        ir_spec = np.asarray(ev["ir_spec"], dtype=float)

        if normalize_ir:
            ir_spec = _normalize_spectrum(ir_spec)

        # --------------------------------------------------------
        # Read CRYSTAL reference modes once.
        # --------------------------------------------------------
        crystal_modes = read_crystal_modes(ref_db, struct)
        freqs_crys = np.asarray(crystal_modes["freqs_cm"], dtype=float)

        # --------------------------------------------------------
        # 1) IR spectrum + frequency correlation.
        # --------------------------------------------------------
        ir_out_png = outdir / f"{struct}_ir_corr_comp.png"

        plot_ir_spectrum_with_frequency_correlation(
            freqs_cm=freqs_model,
            intensities=intensities,
            nu_grid=nu_grid,
            ir_spec=ir_spec,
            structure=struct,
            crystal_freqs_cm=freqs_crys,
            crystal_db_path=ref_db,
            outfile=ir_out_png,
            functional=FUNCTIONAL[struct],
        )

        ir_out_pdf = ir_out_png.with_suffix(".pdf")

        # --------------------------------------------------------
        # 2) Cached mode-overlap plot.
        # This reads the mode_matching group from ref_db.
        # No mode matching is recomputed here.
        # --------------------------------------------------------
        mode_data = read_mode_overlap(
            ref_db,
            structure=struct,
            run_id=best["run_id"],
            dataset_split=best["dataset_split"],
            sweep_id=best["sweep_id"],
        )

        mode_out_png = outdir / f"{struct}_mode_overlap_combined.png"

        plot_combined_overlap_heatmaps(
            overlap_matrix=mode_data["overlap_cut"],
            group_overlap_matrix=mode_data["group_overlap_matrix"],
            group_matches=mode_data["group_matches"],
            freqs_ref=freqs_crys,
            freqs_test=freqs_model,
            skip_first=skip_first,
            outfile=mode_out_png,
        )

        mode_out_pdf = mode_out_png.with_suffix(".pdf")

        hp_label = best.get("hyperparam_label", "")
        if not hp_label:
            hp_label = format_hyperparams(best.get("hyperparameters", {}))

        written[struct] = {
            "structure": struct,
            "metric": metric,
            "metric_value": best["metric_value"],
            "global_rank": best_rank,
            "dataset_split": best["dataset_split"],
            "sweep_id": best["sweep_id"],
            "run_id": best["run_id"],
            "hyperparameters": best.get("hyperparameters", {}),
            "hyperparam_label": hp_label,
            "ir_corr_comp_pdf": ir_out_pdf,
            "mode_overlap_combined_pdf": mode_out_pdf,
            "outdir": outdir,
        }

        print()
        print(f"Best model for {struct}")
        print("-" * 80)
        print(f"rank          : {best_rank}")
        print(f"{metric:<14}: {best['metric_value']:.6g}")
        print(f"hyperparams   : {hp_label}")
        print(f"dataset split : {best['dataset_split']}")
        print(f"sweep_id      : {best['sweep_id']}")
        print(f"run_id        : {best['run_id']}")
        print(f"saved         : {ir_out_pdf}")
        print(f"saved         : {mode_out_pdf}")

    if not written:
        raise RuntimeError(
            f"No figures were written for structure '{structure}'. "
            "Check whether cached evaluations exist for the requested structure(s)."
        )

    return written


def plot_hse_pbe_phonon_grid(
    *,
    ref_db: Path,
    outdir: Path,
    outfile_stem: str,
    style_path: Path | None,
) -> None:
    apply_style(style_path)

    colors = CMAP([0.2, 0.65])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 10),
        # sharey=True,
        # sharex=True,
        constrained_layout=True,
    )
    i = 0
    abcdlabel = ['a)', 'b)', 'c)', 'd)']
    for ax, (hse_name, pbe_name, title) in zip(axes.ravel(), STRUCTURE_PAIRS):
        hse_modes = read_crystal_modes(ref_db, hse_name)['freqs_cm']
        pbe_modes = read_crystal_modes(ref_db, pbe_name)['freqs_cm']

        ax.scatter(
            hse_modes,
            pbe_modes,
            s=180,
            marker='d',
            alpha=0.66,
            edgecolors=CMAP([0.3]),
            linewidths = 0.75,
            color = CMAP([0.8]),
            label=r"$\Gamma$-Mode correlation",
        )
        f_min = min(np.min(hse_modes), np.min(pbe_modes))
        f_max = max(np.max(hse_modes), np.max(pbe_modes))
        pad = 0.05 * (f_max - f_min) if f_max > f_min else 10.0
        f_min -= pad
        f_max += pad
        ax.plot(
            [f_min, f_max],
            [f_min, f_max],
            ls="--",
            lw=0.7,
            color="grey",
        )
        title = f'{abcdlabel[i]} '+title
        ax.text(
            0.05,
            0.95,
            title,
            transform=ax.transAxes,
            va='top', 
            ha='left',
            size=22,
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.85), 
            )
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(f_min, f_max)
        ax.set_aspect("equal", adjustable="box")
        # ax.legend()
        i += 1

    fig.supylabel(r'PBEsol Modes in cm$^{-1}$')
    fig.supxlabel(r'HSEsol Modes in cm$^{-1}$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[0:1], 
        labels[0:1],
        loc="lower center", 
        ncol=1, 
        bbox_to_anchor=(0.5, 0.98), 
        fontsize=22, 
        framealpha=0.9, 
        frameon=True,

    )

    png = outdir / f"{outfile_stem}.png"
    pdf = outdir / f"{outfile_stem}.pdf"

    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_modes(
    *,
    ref_db: Path,
    outdir: Path,
    outfile_stem: str,
    style_path: Path | None,
    skip_first: int = 3
) -> None:
    apply_style(style_path)

    colors = CMAP([0.2, 0.65])

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 9),
        constrained_layout=True,
    )

    abclabels = ['a', 'b', 'c', 'd']
    for ax, (struct_hse, struct_pbe, label), alph in zip(axs.ravel(), STRUCTURE_PAIRS, abclabels):
 
        hse = read_crystal_modes(ref_db, struct_hse)
        pbe = read_crystal_modes(ref_db, struct_pbe)
        
        freqs_ref = hse['freqs_cm']
        freqs_test = pbe['freqs_cm']

        overlap_matrix = compare_mode_sets(
            freqs_crys=hse['freqs_cm'],
            evecs_crys=hse['eigvecs_mw'],
            mace_modes=pbe,
            mode='return_overlap_matrix',
            heatmap_outfile=None
        )
        overlap_matrix = np.asarray(overlap_matrix, dtype=float)
        im = ax.imshow(
            overlap_matrix.T,
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
            cmap=_slice_CMAP('managua_r')
        )

        ax.text(
            0.03,
            0.95,
            f"{alph}) Mode Overlap {label}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            size=18,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # x-axis: HSEsol frequencies
        xlabels = [f"{f:.0f}" for f in freqs_ref[skip_first:]]
        step_x = max(1, len(xlabels) // 12)
        ax.set_xticks(np.arange(0, len(xlabels), step_x))
        ax.set_xticklabels(xlabels[::step_x], rotation=90)

        # y-axis: PBEsol frequencies
        ylabels = [f"{f:.0f}" for f in freqs_test[skip_first:]]
        step_y = max(1, len(ylabels) // 12)
        ax.set_yticks(np.arange(0, len(ylabels), step_y))
        ax.set_yticklabels(ylabels[::step_y])

        #ax.set_xlabel(r"HSEsol modes in cm$^{-1}$")
        #ax.xaxis.set_label_coords(0.5, -0.15)
        #ax.set_ylabel(r"PBEsol modes in cm$^{-1}$")

    cbar = fig.colorbar(im, ax=axs, shrink=0.6)
    cbar.set_label("Overlap")

    fig.supylabel(r"PBEsol Modes in cm$^{-1}$")
    fig.supxlabel(r"HSEsol Modes in cm$^{-1}$")

    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{outfile_stem}.png"
    pdf = outdir / f"{outfile_stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")


def plot_phonon_frequency_evolution_pair(
    *,
    ref_db: Path,
    structure_base: str,
    outdir: Path,
    outfile_stem: str | None = None,
    metric: str = "composite_score",
    selection: str = "spaced",
    top_n: int | None = None,
    modes: str = "matched_modes",
    y_axis: str = "rank_percentile",
    color_by: str = "abs_delta_cm1",
    hparam_filters: dict[str, object] | None = None,
    skip_first: int = 3,
    ir_intensity_tolerance_cm1: float = 2.0,
    show_error_bands: bool = True,
    color_percentile: float = 98.0,
    style_path: Path | None = None,
    scale: float = 1.0,
) -> None:
    """
    Plot model-frequency evolution against model quality for HSEsol/PBEsol.

    Main use:
        x-axis  : model-predicted matched frequency
        y-axis  : rank percentile, rank, filtered rank, or metric
        vlines  : CRYSTAL reference frequencies
        color   : |Δν|, 1-overlap, overlap, or intensity error
        size    : normalized reference IR intensity

    modes
    -----
    matched_modes:
        Uses mode_matching/mode_matches.
        Valid color_by:
            abs_delta_cm1
            one_minus_overlap
            overlap

    ir_active:
        Uses ir_matching/matched_peaks.
        Valid color_by:
            abs_delta_cm1
            intensity_abs_error
    """
    apply_style(style_path)

    ref_db = Path(ref_db)
    outdir = Path(outdir)

    if selection not in {"top", "spaced"}:
        raise ValueError("selection must be 'top' or 'spaced'.")

    if modes not in {"matched_modes", "ir_active"}:
        raise ValueError("modes must be 'matched_modes' or 'ir_active'.")

    hse_name, pbe_name, structure_label = _resolve_structure_pair(structure_base)
    structures = [hse_name, pbe_name]

    panel_data = []
    selected_info = []

    for struct in structures:
        rows_all = list_model_evaluations_for_structure(
            ref_db,
            struct,
            metric=metric,
            require_ir_spectrum=True,
            require_metric=True,
            sort=True,
        )

        if not rows_all:
            raise ValueError(
                f"No cached model evaluations with metric={metric} found for {struct}."
            )

        rows = _filter_rows_by_hyperparams(rows_all, hparam_filters)

        if not rows:
            raise ValueError(
                f"No rows left after filtering for {struct}. Filters: {hparam_filters}"
            )

        if top_n is None:
            selected = rows

        elif top_n < 1:
            raise ValueError("top_n must be >= 1 when provided.")

        elif top_n >= len(rows):
            selected = rows

        elif selection == "top":
            selected = rows[:top_n]

        elif selection == "spaced":
            selected = _select_rank_spaced(rows, top_n)

        else:
            raise ValueError(
                f"Unknown selection mode: {selection}. "
                "Use 'top' or 'spaced'."
            )

        points = _collect_frequency_evolution_points(
            ref_db=ref_db,
            structure=struct,
            rows_all=rows_all,
            selected_rows=selected,
            rows_filtered=rows,
            metric=metric,
            modes=modes,
            y_axis=y_axis,
            color_by=color_by,
            skip_first=skip_first,
            ir_intensity_tolerance_cm1=ir_intensity_tolerance_cm1,
        )

        ref_lines, ref_line_intensities = _reference_lines_for_frequency_evolution(
            ref_db=ref_db,
            structure=struct,
            modes=modes,
            skip_first=skip_first,
        )

        panel_data.append({
            "structure": struct,
            "rows_all": rows_all,
            "rows_filtered": rows,
            "selected": selected,
            "points": points,
            "ref_lines": ref_lines,
            "ref_line_intensities": ref_line_intensities,
        })

        selected_info.append((struct, rows_all, rows, selected))

    # Shared color scale.
    all_c = np.concatenate([
        p["points"]["c"]
        for p in panel_data
        if p["points"]["c"].size
    ])

    if all_c.size == 0:
        raise ValueError("No points collected. Check cached matching data.")

    cmin = float(np.nanmin(all_c))
    cmax = float(np.nanpercentile(all_c, color_percentile))

    if not np.isfinite(cmin):
        cmin = 0.0

    if not np.isfinite(cmax) or cmax <= cmin:
        cmax = float(np.nanmax(all_c))

    if not np.isfinite(cmax) or cmax <= cmin:
        cmax = cmin + 1.0

    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.0 * scale, 6.2 * scale),
        sharey=True,
        constrained_layout=True,
    )

    last_scatter = None

    for ax, pdata, functional_label in zip(axes, panel_data, ["HSEsol", "PBEsol"]):
        points = pdata["points"]

        _draw_reference_frequency_lines(
            ax,
            ref_freqs=pdata["ref_lines"],
            ref_intensities=pdata["ref_line_intensities"],
        )

        if show_error_bands:
            _draw_frequency_error_bands(
                ax,
                points=points,
            )

        sizes = _marker_sizes_from_intensity(points["intensity"])

        last_scatter = ax.scatter(
            points["x"],
            points["y"],
            c=points["c"],
            s=sizes,
            cmap=CMAP,
            norm=norm,
            marker="o",
            alpha=0.72,
            edgecolors="black",
            linewidths=0.25,
            zorder=2,
        )

        title = f"{structure_label} {functional_label}"
        if pdata["structure"] in FUNCTIONAL:
            title = FUNCTIONAL[pdata["structure"]]

        ax.text(
            0.03,
            1.02,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        # x-limits from reference and model frequencies.
        x_candidates = []
        if pdata["ref_lines"].size:
            x_candidates.append(pdata["ref_lines"])
        if points["x"].size:
            x_candidates.append(points["x"])

        if x_candidates:
            xx = np.concatenate(x_candidates)
            xx = xx[np.isfinite(xx)]
            if xx.size:
                xmin = float(np.min(xx))
                xmax = float(np.max(xx))
                pad = 0.05 * (xmax - xmin) if xmax > xmin else 20.0
                ax.set_xlim(max(0.0, xmin - pad), xmax + pad)

        
        ax.margins(x=0.01)

    fig.supxlabel(r"Matched model frequency in cm$^{-1}$")

    if y_axis == "rank_percentile":
        axes[0].set_ylabel("Model rank percentile, 0 = best")
    elif y_axis == "rank":
        axes[0].set_ylabel("Global model rank, 1 = best")
    elif y_axis == "filtered_rank":
        axes[0].set_ylabel("Filtered model rank, 1 = best")
    elif y_axis == "metric":
        axes[0].set_ylabel(metric)

    if color_by == "abs_delta_cm1":
        cbar_label = r"$|\Delta \nu|$ in cm$^{-1}$"
    elif color_by == "one_minus_overlap":
        cbar_label = r"$1 -$ mode overlap"
    elif color_by == "overlap":
        cbar_label = "Mode overlap"
    elif color_by == "intensity_abs_error":
        cbar_label = "Abs. norm. intensity error"
    else:
        cbar_label = color_by

    cbar = fig.colorbar(
        last_scatter, 
        ax=axes, 
        pad = 0.015,
        anchor=(0.0, 0.0),
        shrink=0.775,
        )
    cbar.set_label(cbar_label)

    # Size legend for reference intensity.
    handles = []
    labels = []
    legend_values = np.array([0.0, 0.1, 1.0])
    legend_sizes = _marker_sizes_from_intensity(legend_values)

    for size, label in zip(
        legend_sizes,
        ["IR inactive/weak", "medium IR", "strong IR"],
    ):
        handles.append(
            axes[1].scatter(
                [],
                [],
                s=size,
                facecolors="none",
                edgecolors="black",
                linewidths=0.6,
            )
        )
        labels.append(label)

    fig.legend(
        handles,
        labels,
        title="Reference intensity",
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        fontsize=11,
        title_fontsize=12,
        bbox_to_anchor=(0.92, 0.9625)
    )

    outdir.mkdir(parents=True, exist_ok=True)

    if outfile_stem is None:
        filter_suffix = ""
        if hparam_filters:
            filter_suffix = "_" + "_".join(
                f"{k}-{v}" for k, v in hparam_filters.items()
            )
            filter_suffix = filter_suffix.replace("/", "_")

        n_label = "all" if top_n is None else str(top_n)
        outfile_stem = (
            f"{structure_base}_{modes}_{selection}_{top_n}_"
            f"{metric}_{y_axis}_{color_by}{filter_suffix}_frequency_evolution"
        )

    png = outdir / f"{outfile_stem}.png"
    pdf = outdir / f"{outfile_stem}.pdf"

    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    print()
    print(f"Frequency evolution plot: {structure_base}")
    print(f"modes      : {modes}")
    print(f"metric     : {metric}")
    print(f"selection  : {selection}")
    print(f"top_n      : {'all' if top_n is None else top_n}")
    print(f"y_axis     : {y_axis}")
    print(f"color_by   : {color_by}")
    if hparam_filters:
        print(f"filters    : {hparam_filters}")

    for struct, rows_all, rows, selected in selected_info:
        print()
        print(f"{struct}: selected {len(selected)} runs")
        for row in selected[:10]:
            global_rank = rows_all.index(row) + 1
            filtered_rank = rows.index(row) + 1
            hp_label = row.get("hyperparam_label", "")
            if not hp_label:
                hp_label = format_hyperparams(row.get("hyperparameters", {}))

            print(
                f"  global rank {global_rank:>4d} | "
                f"filtered rank {filtered_rank:>4d} | "
                f"{metric} {row['metric_value']:>10.5g} | "
                f"{hp_label}"
            )

        if len(selected) > 10:
            print(f"  ... {len(selected) - 10} more")



###########################################################################
# MAIN
###########################################################################


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ref-db", type=Path, default=DEFAULT_REF_DB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project-level plotting CLI for CRYSTALdataGen."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_mode_grid = sub.add_parser(
        "mode-overlap-grid",
        help="Plot HSEsol vs PBEsol CRYSTAL reference modes in a NxN grid."
    )
    add_common_args(p_mode_grid)
    p_mode_grid.add_argument("--outfile-stem", default="hse_pbe_mode_overlap")

    p_phonon_grid = sub.add_parser(
        "phonon-grid",
        help="Plot HSEsol vs PBEsol CRYSTAL reference phonons in a 2x2 grid."
    )
    add_common_args(p_phonon_grid)
    p_phonon_grid.add_argument("--outfile-stem", default="hse_pbe_phonon_grid")


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

    p_best_results = sub.add_parser(
        "best-result-figures",
        help=(
            "Plot IR correlation and combined mode-overlap figures for the "
            "best cached model of a structure. By default also plots the "
            "paired HSEsol/PBEsol functional if available."
        ),
    )
    add_common_args(p_best_results)
    p_best_results.add_argument("--structure", required=True)
    p_best_results.add_argument("--metric", default="composite_score")
    p_best_results.add_argument(
        "--outdir-root",
        type=Path,
        default=Path("/home/jha/jha/python_scripts/master_thesis/figures/results"),
    )
    p_best_results.add_argument(
        "--single-functional",
        action="store_true",
        help="Only plot the requested structure, not the paired functional.",
    )
    p_best_results.add_argument(
        "--hparam",
        action="append",
        default=None,
        help=(
            "Filter by hyperparameter before selecting the best model. "
            "Examples: --hparam rmax=4 --hparam fw=75 --hparam seed=2"
        ),
    )
    p_best_results.add_argument("--skip-first", type=int, default=3)
    p_best_results.add_argument("--degeneracy-tol", type=float, default=0.5)
    p_best_results.add_argument(
        "--no-normalize-ir",
        action="store_true",
        help="Use stored model spectrum amplitudes without renormalizing.",
    )

    p_freq_evo = sub.add_parser(
        "frequency-evolution",
        help=(
            "Plot matched phonon-frequency evolution over ranked models, "
            "with HSEsol and PBEsol next to each other."
        ),
    )
    add_common_args(p_freq_evo)
    p_freq_evo.add_argument("--structure", required=True)
    p_freq_evo.add_argument("--outfile-stem", default=None)
    p_freq_evo.add_argument("--metric", default="composite_score")
    p_freq_evo.add_argument(
        "--selection",
        choices=["top", "spaced"],
        default="spaced",
    )
    p_freq_evo.add_argument("--top-n", type=int, default=None)
    p_freq_evo.add_argument(
        "--modes",
        choices=["matched_modes", "ir_active"],
        default="matched_modes",
    )
    p_freq_evo.add_argument(
        "--y-axis",
        choices=["rank", "filtered_rank", "rank_percentile", "metric"],
        default="rank_percentile",
    )
    p_freq_evo.add_argument(
        "--color-by",
        choices=[
            "abs_delta_cm1",
            "one_minus_overlap",
            "overlap",
            "intensity_abs_error",
        ],
        default="abs_delta_cm1",
    )
    p_freq_evo.add_argument(
        "--hparam",
        action="append",
        default=None,
        help=(
            "Filter by hyperparameter. Can be given multiple times. "
            "Examples: --hparam rmax=4 --hparam fw=75 --hparam seed=2"
        ),
    )
    p_freq_evo.add_argument("--skip-first", type=int, default=3)
    p_freq_evo.add_argument("--ir-intensity-tolerance", type=float, default=2.0)
    p_freq_evo.add_argument(
        "--no-error-bands",
        action="store_true",
        help="Disable median frequency-error bands around reference lines.",
    )
    p_freq_evo.add_argument("--color-percentile", type=float, default=98.0)

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

    elif args.command == "phonon-grid":
        plot_hse_pbe_phonon_grid(
            ref_db=args.ref_db,
            outdir=args.outdir,
            outfile_stem=args.outfile_stem,
            style_path=args.style,
        )

    elif args.command == "mode-overlap-grid":
        plot_modes(
            ref_db=args.ref_db,
            outdir=args.outdir,
            outfile_stem=args.outfile_stem,
            style_path=args.style,
        )

    elif args.command == "best-result-figures":
        hparam_filters = _parse_hparam_filters(args.hparam)

        plot_best_model_result_figures_for_structure(
            ref_db=args.ref_db,
            structure=args.structure,
            outdir_root=args.outdir_root,
            metric=args.metric,
            include_functional_pair=not args.single_functional,
            hparam_filters=hparam_filters,
            style_path=args.style,
            normalize_ir=not args.no_normalize_ir,
            skip_first=args.skip_first,
            degeneracy_tol=args.degeneracy_tol,
        )

    elif args.command == "frequency-evolution":
        hparam_filters = _parse_hparam_filters(args.hparam)

        plot_phonon_frequency_evolution_pair(
            ref_db=args.ref_db,
            structure_base=args.structure,
            outdir=args.outdir,
            outfile_stem=args.outfile_stem,
            metric=args.metric,
            selection=args.selection,
            top_n=args.top_n,
            modes=args.modes,
            y_axis=args.y_axis,
            color_by=args.color_by,
            hparam_filters=hparam_filters,
            skip_first=args.skip_first,
            ir_intensity_tolerance_cm1=args.ir_intensity_tolerance,
            show_error_bands=not args.no_error_bands,
            color_percentile=args.color_percentile,
            style_path=args.style,
        )

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

