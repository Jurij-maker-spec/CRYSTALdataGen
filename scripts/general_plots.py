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
    list_model_evaluations_for_structure,
    format_hyperparams,
)
from util.plotting import restore_degeneracies, perform_KDE
from util.mode_matching import (
    compare_mode_sets,
    _slice_CMAP
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
    degeneracy_tol: float = 0.5,
    skip_missing: bool = True,
) -> dict[str, dict]:
    """
    Plot best-model result figures for one structure, optionally for both
    HSEsol and PBEsol variants.

    For each selected structure, this creates:

        <structure>_ir_corr_comp.pdf
        <structure>_mode_overlap_combined.pdf

    and saves them into a pooled per-base-structure directory, e.g.

        /home/jha/jha/python_scripts/master_thesis/figures/results/AlN/

    Parameters
    ----------
    ref_db
        Path to ref_db.h5.

    structure
        Structure name, e.g. "AlN", "AlN_PBE", "SiO2", "TiO2_rutil".

    outdir_root
        Root output directory. The function creates one subdirectory per
        base structure.

    metric
        Ranking metric. Lower is assumed to be better, consistent with
        list_model_evaluations_for_structure(..., sort=True).

    include_functional_pair
        If True and `structure` belongs to STRUCTURE_PAIRS, plot both
        HSEsol and PBEsol variants into the same base-structure directory.

    hparam_filters
        Optional filters, same logic as plot_top_N_of_a_structure(), e.g.
        {"r_max": 4.0, "forces_weight": 75}.

    style_path
        Matplotlib style file.

    normalize_ir
        Normalize the cached model IR spectrum before plotting.

    skip_first
        Number of acoustic modes to skip in mode-overlap plots.

    degeneracy_tol
        Frequency tolerance in cm^-1 for degenerate-group mode matching.

    skip_missing
        If True, missing PBE/HSEsol variants are skipped with a warning.
        If False, missing data raises an error.

    Returns
    -------
    written
        Dict keyed by plotted structure, containing selected run metadata
        and output paths.
    """
    from util.plotting import plot_ir_spectrum_with_frequency_correlation
    from util.mode_matching import plot_combined_overlap_heatmaps

    apply_style(style_path)

    ref_db = Path(ref_db)
    outdir_root = Path(outdir_root)

    def _resolve_structure_family(structure_name: str):
        """
        Return base structure and list of structures to plot.

        Example
        -------
        structure_name="AlN"     -> ("AlN", ["AlN", "AlN_PBE"])
        structure_name="AlN_PBE" -> ("AlN", ["AlN", "AlN_PBE"])
        """
        for hse_name, pbe_name, _label in STRUCTURE_PAIRS:
            if structure_name in {hse_name, pbe_name}:
                if include_functional_pair:
                    return hse_name, [hse_name, pbe_name]
                return hse_name, [structure_name]

        # Fallback for structures not listed in STRUCTURE_PAIRS.
        if structure_name.endswith("_PBE"):
            base_name = structure_name[:-4]
        else:
            base_name = structure_name

        return base_name, [structure_name]

    def _first_existing(mapping: dict, keys: list[str], *, required: bool = True):
        for key in keys:
            if key in mapping:
                value = mapping[key]
                if value is not None:
                    return value

        if required:
            raise KeyError(
                "None of the expected keys were found. "
                f"Tried: {keys}. Available keys: {sorted(mapping.keys())}"
            )

        return None

    base_structure, structures_to_plot = _resolve_structure_family(structure)
    outdir = outdir_root / base_structure
    outdir.mkdir(parents=True, exist_ok=True)

    written: dict[str, dict] = {}

    for struct in structures_to_plot:
        # --------------------------------------------------------
        # Rank cached model evaluations and select the best one.
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

        ev = read_model_evaluation(
            ref_db,
            struct,
            run_id=best["run_id"],
            dataset_split=best["dataset_split"],
            sweep_id=best["sweep_id"],
        )

        # --------------------------------------------------------
        # Required cached model arrays.
        # --------------------------------------------------------
        freqs_model = np.asarray(
            _first_existing(
                ev,
                ["freqs_cm", "frequencies_cm", "frequencies_cm1"],
            ),
            dtype=float,
        )

        eigvecs_model = np.asarray(
            _first_existing(
                ev,
                ["eigvecs_mw", "eigenvectors_mw", "modes_mw"],
            ),
            dtype=float,
        )

        nu_grid = np.asarray(
            _first_existing(
                ev,
                ["nu_grid_cm1", "nu_grid", "wavenumber_grid_cm1"],
            ),
            dtype=float,
        )

        ir_spec = np.asarray(
            _first_existing(
                ev,
                ["ir_spec", "ir_spectrum", "spectrum"],
            ),
            dtype=float,
        )

        intensities = _first_existing(
            ev,
            ["intensities", "ir_intensities", "intensities_km_mol"],
            required=False,
        )
        if intensities is None:
            intensities = np.zeros_like(freqs_model)
        intensities = np.asarray(intensities, dtype=float)

        if normalize_ir:
            ir_spec = _normalize_spectrum(ir_spec)

        # --------------------------------------------------------
        # Read CRYSTAL reference modes once. Used for both plots:
        # frequency correlation and mode-overlap.
        # --------------------------------------------------------
        crystal_modes = read_crystal_modes(ref_db, struct)
        freqs_crys = np.asarray(crystal_modes["freqs_cm"], dtype=float)
        eigvecs_crys = np.asarray(crystal_modes["eigvecs_mw"], dtype=float)

        # --------------------------------------------------------
        # 1) IR spectrum + frequency-correlation plot.
        # Passing a .png path creates both .png and .pdf in your
        # current util.plotting implementation.
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
            functional = FUNCTIONAL[struct]
        )

        ir_out_pdf = ir_out_png.with_suffix(".pdf")

        # --------------------------------------------------------
        # 2) Combined mode-overlap plot.
        # We call compare_mode_sets without heatmap_outfile to avoid
        # writing the separate single/group heatmaps, then write only
        # the combined figure.
        # --------------------------------------------------------
        mace_modes = {
            "freqs_cm": freqs_model,
            "eigvecs_mw": eigvecs_model,
        }

        mode_result = compare_mode_sets(
            freqs_crys=freqs_crys,
            evecs_crys=eigvecs_crys,
            mace_modes=mace_modes,
            skip_first=skip_first,
            degeneracy_tol=degeneracy_tol,
            heatmap_outfile=None,
            title=f"{struct}: CRYSTAL vs MACELES mode overlap",
        )

        mode_out_png = outdir / f"{struct}_mode_overlap_combined.png"

        plot_combined_overlap_heatmaps(
            overlap_matrix=mode_result["overlap_cut"],
            group_overlap_matrix=mode_result["group_overlap_matrix"],
            group_matches=mode_result["subgroups"],
            freqs_ref=freqs_crys,
            freqs_test=freqs_model,
            skip_first=skip_first,
            outfile=mode_out_png,
            functional = FUNCTIONAL[struct]
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
        figsize=(9, 9),
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
            s=160,
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

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

