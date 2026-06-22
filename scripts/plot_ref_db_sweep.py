#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from explore_ref_db_sweep import (
    split_matches_structure,
    iter_evaluation_runs,
    make_analysis_row,
    make_success_threshold,
    safe_float,
    safe_int,
)

REF_PATH = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5")
mpl.style.use('/home/jha/jha/python_scripts/CRYSTALdataGen/util/style.mplstyle')
CMAP = plt.get_cmap("managua")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
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
MARKER_SIZE_MIN = 45.0
MARKER_SIZE_MAX = 250.0
HATCH_SEQUENCE = ["", "/", "x", "\\\\", "-", "+", "o", "O", ".", "*"]
PREFERRED_FORCE_HATCHES = {
    50.0: "",
    75.0: "/////",
    100.0: "xxxxx",
}

def make_force_hatch_map(force_weights):
    force_weights = sorted(float(fw) for fw in force_weights)

    hatch_map = {}
    for i, fw in enumerate(force_weights):
        if fw in PREFERRED_FORCE_HATCHES:
            hatch_map[fw] = PREFERRED_FORCE_HATCHES[fw]
        else:
            hatch_map[fw] = HATCH_SEQUENCE[i % len(HATCH_SEQUENCE)]

    return hatch_map


def add_broken_axis_marks(
    ax_top,
    ax_bottom,
    *,
    slash_size=0.012,
    linewidth=0.9,
    color="black",
    draw_horizontal=True,
    draw_diagonal=True,
):
    """
    Draw visual marks for a broken y-axis.

    draw_horizontal=True:
        draws thin horizontal separator lines at the break.

    draw_diagonal=True:
        draws the standard diagonal slashes at the left/right edges.
    """
    if draw_horizontal:
        ax_top.plot(
            [0.0, 1.0],
            [0.0, 0.0],
            transform=ax_top.transAxes,
            color=color,
            linewidth=linewidth,
            clip_on=False,
        )
        ax_bottom.plot(
            [0.0, 1.0],
            [1.0, 1.0],
            transform=ax_bottom.transAxes,
            color=color,
            linewidth=linewidth,
            clip_on=False,
        )

    if draw_diagonal:
        d = slash_size

        kwargs_top = dict(
            transform=ax_top.transAxes,
            color=color,
            clip_on=False,
            linewidth=linewidth,
        )
        kwargs_bottom = dict(
            transform=ax_bottom.transAxes,
            color=color,
            clip_on=False,
            linewidth=linewidth,
        )

        # bottom edge of upper axis
        ax_top.plot((-d, +d), (-d, +d), **kwargs_top)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_top)

        # top edge of lower axis
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bottom)


def collect_rows(h5, structure: str, include_all_splits: bool = False):
    """
    Collect one plotting row per evaluated model.

    Important: this reuses make_analysis_row() from explore_ref_db_sweep.py,
    therefore DB-stored hyperparameters are preferred over run_id parsing.
    In particular, the training dataset size is read from row["size"].
    """
    rows = []

    for path_split, sweep_id, run_id, run_group in iter_evaluation_runs(h5, structure) or []:
        if not include_all_splits and not split_matches_structure(path_split, structure):
            continue
        if "ranking_metrics" not in run_group:
            continue

        row = make_analysis_row(
            structure=structure,
            path_split=path_split,
            sweep_id=sweep_id,
            run_id=run_id,
            run_group=run_group,
        )
        rows.append(row)

    return rows


def value_array(rows, key):
    values = []
    keep = []

    for row in rows:
        value = safe_float(row.get(key))
        if not np.isfinite(value):
            continue
        values.append(value)
        keep.append(row)

    return np.asarray(values, dtype=float), keep


def require_keys(rows, keys):
    out = []
    for row in rows:
        ok = True
        for key in keys:
            if not np.isfinite(safe_float(row.get(key))):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def marker_for_seed(seed):
    markers = ["v", "o", "d"]
    return markers[int(seed) % len(markers)]


def savefig(fig, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved: {outbase.with_suffix('.png')}")
    print(f"saved: {outbase.with_suffix('.pdf')}")


def _format_numeric_label(value):
    value = safe_float(value)
    if not np.isfinite(value):
        return "--"
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:g}"


def make_train_size_map(rows, *, min_area=MARKER_SIZE_MIN, max_area=MARKER_SIZE_MAX):
    """Map DB train dataset sizes N to marker areas in points^2."""
    train_sizes = sorted({safe_int(r.get("size")) for r in rows if safe_int(r.get("size")) is not None})
    if not train_sizes:
        return [], {}

    if len(train_sizes) == 1:
        return train_sizes, {train_sizes[0]: 0.5 * (min_area + max_area)}

    lo = float(min(train_sizes))
    hi = float(max(train_sizes))
    size_map = {
        n: min_area + (float(n) - lo) / (hi - lo) * (max_area - min_area)
        for n in train_sizes
    }
    return train_sizes, size_map


def make_hatch_map(force_weights):
    """Map force weights to hatch patterns; preserves the requested 50/75/100 convention."""
    hatch_map = {}
    next_idx = 0

    for fw in force_weights:
        fw_key = float(fw)
        matched = None
        for preferred_fw, hatch in PREFERRED_FORCE_HATCHES.items():
            if np.isclose(fw_key, preferred_fw):
                matched = hatch
                break
        if matched is not None:
            hatch_map[fw_key] = matched
            continue

        hatch_map[fw_key] = HATCH_SEQUENCE[next_idx % len(HATCH_SEQUENCE)]
        next_idx += 1

    return hatch_map


def make_style_maps(rows):
    energy_weights = sorted(set(float(r["energy_weight"]) for r in rows))
    seeds = sorted(set(int(r["seed"]) for r in rows))
    force_weights = sorted(set(float(r["forces_weight"]) for r in rows))
    train_sizes = sorted(set(float(r["size"]) for r in rows))

    cmap = CMAP
    e_colors = cmap(np.linspace(0.10, 0.90, len(energy_weights)))

    energy_colors = {
        ew: e_colors[i]
        for i, ew in enumerate(energy_weights)
    }

    size_values = np.asarray(train_sizes, dtype=float)

    if len(size_values) == 1:
        size_areas = np.asarray([120.0])
    else:
        size_areas = np.interp(
            size_values,
            (np.min(size_values), np.max(size_values)),
            (35.0, 270.0),
        )

    size_area_map = {
        size: area
        for size, area in zip(train_sizes, size_areas)
    }

    force_hatches = make_force_hatch_map(force_weights)

    return (
        energy_weights,
        seeds,
        force_weights,
        train_sizes,
        size_area_map,
        energy_colors,
        force_hatches,
    )


def _get_style_values(row, energy_colors, hatch_map, train_size_map):
    ew = float(safe_float(row.get("energy_weight")))
    fw = float(safe_float(row.get("forces_weight")))
    seed = int(safe_int(row.get("seed")))
    train_size = safe_int(row.get("size"))

    return {
        "color": energy_colors[ew],
        "marker": marker_for_seed(seed),
        "area": train_size_map[train_size],
        "hatch": hatch_map.get(fw, ""),
    }


def scatter_styled_rows(ax, rows, x_key, y_key, energy_colors, hatch_map, train_size_map):
    """
    Scatter rows one-by-one so each point can have its own hatch and size.
    Matplotlib scatter accepts one hatch per PathCollection, not a list of hatches.
    """
    for row in rows:
        style = _get_style_values(row, energy_colors, hatch_map, train_size_map)
        ax.scatter(
            [safe_float(row.get(x_key))],
            [safe_float(row.get(y_key))],
            c=[style["color"]],
            marker=style["marker"],
            s=style["area"],
            hatch=style["hatch"],
            hatchcolor='black',
            hatch_linewidth = 0.15,
            alpha=0.65,
            edgecolors="black",
            linewidths=0.45,
        )


def add_style_legends(
    fig,
    seeds,
    energy_weights,
    force_weights,
    train_sizes,
    energy_colors,
    force_hatches,
    size_area_map,
    *,
    # Main plot / legend-column layout
    right_margin=0.74,
    legend_left=0.752,
    legend_bottom=0.10,
    legend_width=0.22,
    legend_height=0.82,
    # Adaptive legend sizing
    base_height=0.070,
    item_height=0.040,
    gap=0.002,
    # Text / box styling
    fontsize=13,
    title_fontsize=14,
    frameon=True,
):
    """
    Add stacked, equally wide legends in a fixed right-side legend column.

    Each legend block has the same width. Its height is computed as

        block_height = base_height + item_height * number_of_entries

    Blocks are filled from the top downward.
    """
    # Reserve space for the legend column.
    fig.subplots_adjust(right=right_margin)

    # Invisible axis used only as a coordinate system for the legends.
    legend_ax = fig.add_axes(
        [legend_left, legend_bottom, legend_width, legend_height]
    )
    legend_ax.set_axis_off()

    y_top = 0.9525
    # y_top = 1.0
    def add_legend_block(handles, title):
        nonlocal y_top

        if len(handles) == 0:
            return
        block_height = base_height + item_height * len(handles)
        y_bottom = y_top - block_height

        # If the legends become too tall, still place them;
        # this makes the problem obvious instead of silently overlapping.
        legend = legend_ax.legend(
            handles=handles,
            title=title,
            loc="upper left",
            bbox_to_anchor=(0.0, y_bottom, 1.0, block_height),
            bbox_transform=legend_ax.transAxes,
            mode="expand",
            frameon=frameon,
            borderaxespad=0.0,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
            handlelength=1.8,
            handletextpad=0.8,
            labelspacing=0.6,
            borderpad=0.5,
        )
        legend_ax.add_artist(legend)
        y_top = y_bottom - gap

    seed_handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=marker_for_seed(seed),
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            markersize=8,
            label=f"seed {int(seed)}",
        )
        for seed in seeds
    ]

    ew_handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markerfacecolor=energy_colors[float(ew)],
            markeredgecolor="black",
            color="black",
            markersize=8,
            label=fr"$w_E={float(ew):g}$",
        )
        for ew in energy_weights
    ]

    fw_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=force_hatches.get(float(fw), ""),
            label=fr"$w_F={float(fw):g}$",
        )
        for fw in force_weights
    ]

    size_handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            markersize=np.sqrt(size_area_map[float(size)]),
            label=fr"$N={int(size)}$",
        )
        for size in train_sizes
    ]

    add_legend_block(seed_handles, "Seed")
    add_legend_block(ew_handles, r"Energy weight")
    add_legend_block(fw_handles, r"Force weight")
    add_legend_block(size_handles, r"Training size")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_landscape(
        rows, 
        structure, 
        outdir, 
        top_fraction=0.10,
        label_pos='top'
        ):
    keys = ["r_max", "composite_score", "energy_weight", "forces_weight", "seed", "size"]
    rows = require_keys(rows, keys)
    if not rows:
        raise RuntimeError("No rows with all required landscape keys.")

    rows = sorted(rows, key=lambda r: safe_float(r.get("composite_score")))
    comp_score = np.asarray([safe_float(r.get("composite_score")) for r in rows], dtype=float)
    cs_min = float(np.min(comp_score))
    cs_max = float(np.max(comp_score))

    top_threshold = make_success_threshold(
        rows,
        metric="composite_score",
        success_quantile=top_fraction,
    )
    top_rows = [r for r in rows if safe_float(r.get("composite_score")) <= top_threshold]

    (
        energy_weights,
        seeds,
        force_weights,
        train_sizes,
        size_area_map,
        energy_colors,
        force_hatches,
    ) = make_style_maps(rows)


    fig, axs = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
        figsize=(7.5, 7.5),
        constrained_layout=False,
    )
    fig.subplots_adjust(hspace=0.06, wspace=0.01, right=0.86)
    ax_top, ax_bottom = axs

    for ax in axs:
        scatter_styled_rows(
            ax,
            rows,
            "r_max",
            "composite_score",
            energy_colors,
            force_hatches,
            size_area_map,
        )

    total_span = max(cs_max - cs_min, 1e-12)
    bottom_span = max(top_threshold - cs_min, 1e-12)
    bottom_pad = max(0.08 * bottom_span, 0.015 * total_span)
    top_pad = max(0.04 * total_span, 1e-12)
    gap = max(0.01 * total_span, 1e-12)

    bottom_upper = top_threshold + bottom_pad
    top_lower = min(top_threshold + gap, cs_max - 0.05 * total_span)
    if top_lower <= bottom_upper:
        top_lower = bottom_upper + gap
    if top_lower >= cs_max:
        top_lower = top_threshold

    ax_bottom.set_ylim(cs_min - bottom_pad, bottom_upper)
    ax_top.set_ylim(top_lower, cs_max + top_pad)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.tick_params(top=False)
    add_broken_axis_marks(ax_top, ax_bottom)

    ax_top.text(
        0.05,
        0.95,
        f"{FUNCTIONAL[structure]}",
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        size=18,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    if label_pos == 'top':
        posi = 0.95
    elif label_pos == 'bottom':
        posi = 0.05
    else:
        label_pos = 'center'
        posi = 0.5
    ax_bottom.text(
        0.05,
        posi,
        f"Top {100.0 * top_fraction:g}\\% models: {len(top_rows)}/{len(rows)}",
        transform=ax_bottom.transAxes,
        va=label_pos,
        ha="left",
        size=15,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax_bottom.set_xlabel(r"Cutoff $r_\mathrm{max}$ in $\AA$")
    fig.supylabel("Composite Score", size=22)

    add_style_legends(
        fig,
        seeds,
        energy_weights,
        force_weights,
        train_sizes,
        energy_colors,
        force_hatches,
        size_area_map,
    )
    
    savefig(fig, outdir / f"{structure}_landscape_rmax_composite")
    plt.close(fig)


def plot_score_decomposition(rows, structure, outdir):
    keys = [
        "score_freq_mae_term",
        "score_intensity_corr_term",
        "energy_weight",
        "forces_weight",
        "seed",
        "size",
    ]
    rows = require_keys(rows, keys)
    if not rows:
        raise RuntimeError("No rows with all required score-decomposition keys.")

    rows = sorted(rows, key=lambda r: safe_float(r.get("score_freq_mae_term")))

    (
        energy_weights,
        seeds,
        force_weights,
        train_sizes,
        size_area_map,
        energy_colors,
        force_hatches,
    ) = make_style_maps(rows)

    fig, ax = plt.subplots(
        figsize=(7.5, 7.5),
        constrained_layout=False,
    )
    fig.subplots_adjust(right=0.86)

    scatter_styled_rows(
        ax,
        rows,
        "score_freq_mae_term",
        "score_intensity_corr_term",
        energy_colors,
        hatch_map,
        train_size_map,
    )

    ax.text(
        0.05,
        0.95,
        f"{structure}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        size=18,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.set_xlabel("Frequency MAE score term")
    ax.set_ylabel("Intensity correlation score term")

    add_style_legends(
        fig,
        seeds,
        energy_weights,
        force_weights,
        train_sizes,
        train_size_map,
        energy_colors,
        hatch_map,
    )

    savefig(fig, outdir / f"{structure}_score_decomposition")
    plt.close(fig)


def plot_physics_quality(rows, structure, outdir):
    overlap_key = None
    for candidate in ["crystal_mode_mean_overlap", "diagonal_overlap_mean"]:
        if any(candidate in r for r in rows):
            overlap_key = candidate
            break

    if overlap_key is None:
        raise KeyError(
            "No mode-overlap metric found. Expected crystal_mode_mean_overlap "
            "or diagonal_overlap_mean."
        )

    keys = [
        overlap_key,
        "freq_mae_ir_cm1",
        "intensity_pearson_r",
        "energy_weight",
        "forces_weight",
        "seed",
        "size",
    ]
    rows = require_keys(rows, keys)
    if not rows:
        raise RuntimeError("No rows with all required physics-quality keys.")

    rows = sorted(rows, key=lambda r: safe_float(r.get("freq_mae_ir_cm1")))

    (
        energy_weights,
        seeds,
        force_weights,
        train_sizes,
        size_area_map,
        energy_colors,
        force_hatches,
    ) = make_style_maps(rows)

    fig, ax = plt.subplots(
        figsize=(7.5, 7.5),
        constrained_layout=False,
    )
    fig.subplots_adjust(right=0.86)

    scatter_styled_rows(
        ax,
        rows,
        overlap_key,
        "freq_mae_ir_cm1",
        energy_colors,
        hatch_map,
        train_size_map,
    )

    ax.text(
        0.05,
        0.95,
        f"{structure}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        size=18,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.set_xlabel(overlap_key)
    ax.set_ylabel(r"IR frequency MAE in cm$^{-1}$")

    add_style_legends(
        fig,
        seeds,
        energy_weights,
        force_weights,
        train_sizes,
        energy_colors,
        force_hatches,
        size_area_map,
    )

    savefig(fig, outdir / f"{structure}_physics_quality")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot ref_db sweep analysis for one structure."
    )
    parser.add_argument("--ref_db", type=Path, default=REF_PATH)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--outdir", type=Path, default=Path("plots/"))
    parser.add_argument(
        "--plot",
        choices=["all", "landscape", "decomposition", "physics"],
        default="all",
    )
    parser.add_argument(
        "--include_all_splits",
        action="store_true",
        help="Include e.g. SiO2_PBE splits when plotting SiO2.",
    )
    parser.add_argument(
        "--top_fraction",
        type=float,
        default=0.10,
        help="Lower-score fraction shown in the lower broken y-axis panel of the landscape plot.",
    )
    parser.add_argument(
        "--label-pos",
        type=str,
        default='bottom',
        help='Position of the lower plot text label'
    )
    args = parser.parse_args()

    with h5py.File(args.ref_db, "r") as h5:
        rows = collect_rows(
            h5,
            args.structure,
            include_all_splits=args.include_all_splits,
        )

    if not rows:
        raise RuntimeError(f"No usable model rows found for {args.structure}")

    print(f"rows used: {len(rows)}")
    args.outdir = args.outdir / f"{args.structure}"

    if args.plot in {"all", "landscape"}:
        plot_landscape(rows, args.structure, args.outdir, top_fraction=args.top_fraction, label_pos=args.label_pos)

    if args.plot in {"all", "decomposition"}:
        plot_score_decomposition(rows, args.structure, args.outdir)

    if args.plot in {"all", "physics"}:
        plot_physics_quality(rows, args.structure, args.outdir)


if __name__ == "__main__":
    main()
