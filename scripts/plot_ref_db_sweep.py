#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from explore_ref_db_sweep import (
    parse_run_id,
    split_matches_structure,
    iter_evaluation_runs,
)

REF_PATH = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/data/ref_db.h5")
mpl.style.use('/home/jha/jha/python_scripts/CRYSTALdataGen/util/style.mplstyle')
CMAP = plt.get_cmap("managua")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def collect_rows(h5, structure: str, include_all_splits: bool = False):
    rows = []

    for split, sweep_id, run_id, run_group in iter_evaluation_runs(h5, structure) or []:
        if not include_all_splits and not split_matches_structure(split, structure):
            continue

        params = parse_run_id(run_id)

        if "ranking_metrics" not in run_group:
            continue

        metrics = dict(run_group["ranking_metrics"].attrs)

        row = {
            "split": split,
            "sweep_id": sweep_id,
            "run_id": run_id,
            **params,
        }

        for key, value in metrics.items():
            try:
                row[key] = float(value)
            except Exception:
                pass

        rows.append(row)

    return rows


def value_array(rows, key):
    values = []
    keep = []

    for row in rows:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
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
            if key not in row:
                ok = False
                break
            try:
                if not np.isfinite(float(row[key])):
                    ok = False
                    break
            except Exception:
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


def make_style_maps(rows):
    energy_weights = sorted(set(float(r["energy_weight"]) for r in rows))
    seeds = sorted(set(int(r["seed"]) for r in rows))
    force_weights = sorted(set(float(r["forces_weight"]) for r in rows))

    size_map = {
        fw: size
        for fw, size in zip(force_weights, np.linspace(10, 200, len(force_weights)))
    }

    cmap = CMAP
    e_colors = cmap(np.linspace(0, 1, len(energy_weights)))

    energy_colors = {
        ew: e_colors[i]
        for i, ew in enumerate(energy_weights)
    }

    return energy_weights, seeds, force_weights, size_map, energy_colors


def add_style_legends(fig, seeds, energy_weights, force_weights, size_map, energy_colors):
    seed_handles = [
        Line2D(
            [], [],
            linestyle="",
            marker=marker_for_seed(seed),
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            markersize=8,
            label=f"seed {seed}",
        )
        for seed in seeds
    ]

    ew_handles = [
        Line2D(
            [], [],
            linestyle="",
            marker="o",
            markerfacecolor=energy_colors[ew],
            markeredgecolor="black",
            color="black",
            markersize=8,
            label=f"ew={ew:g}",
        )
        for ew in energy_weights
    ]

    fw_handles = [
        Line2D(
            [], [],
            linestyle="",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            markersize=np.sqrt(size_map[fw]),
            label=f"fw={fw:g}",
        )
        for fw in force_weights
    ]

    legend_seed = fig.legend(
        handles=seed_handles,
        title="Seed",
        loc="upper left",
        bbox_to_anchor=(0.9, 0.89),
    )
    fig.add_artist(legend_seed)

    legend_ew = fig.legend(
        handles=ew_handles,
        title="E-weight",
        loc="upper left",
        bbox_to_anchor=(0.9, 0.67),
    )
    fig.add_artist(legend_ew)

    fig.legend(
        handles=fw_handles,
        title="F-weight",
        loc="upper left",
        bbox_to_anchor=(0.9, 0.45),
    )

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_landscape(rows, structure, outdir):
    keys = ["r_max", "composite_score", "energy_weight", "forces_weight", "seed"]
    rows = require_keys(rows, keys)
    rows = sorted(rows, key=lambda r: r["composite_score"])

    comp_score = np.array([r['composite_score'] for r in rows])
    cs_max = comp_score.max()
    cs_min = comp_score.min()
    energy_weights = sorted(set(float(r["energy_weight"]) for r in rows))
    seeds = sorted(set(int(r["seed"]) for r in rows))
    force_weights = sorted(set(float(r["forces_weight"]) for r in rows))

    size_map = {
        fw: size for fw, size in zip(force_weights, np.linspace(10, 200, len(force_weights)))
    }

    cmap = CMAP
    e_colors = cmap([0, 0.5, 1.0])
    energy_colors = {
        ew: e_colors[i]
        for i, ew in enumerate(energy_weights)
    }

    fig, axs = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1]},
        figsize=(7.5,7.5),
        constrained_layout=False
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.01)
    
    (ax_top, ax_bottom) = axs
    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]

        x = np.asarray([r["r_max"] for r in sub], dtype=float)
        y = np.asarray([r["composite_score"] for r in sub], dtype=float)

        colors = [energy_colors[float(r["energy_weight"])] for r in sub]
        sizes = [size_map[float(r["forces_weight"])] for r in sub]
        for ax in axs:
            ax.scatter(
                x,
                y,
                c=colors,
                marker=marker_for_seed(seed),
                s=sizes,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.4,
            )

    seed_handles = [
        Line2D(
            [], [],
            linestyle="",
            marker=marker_for_seed(seed),
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            markersize=8,
            label=f"seed {seed}",
        )
        for seed in seeds
    ]
    ew_handles = [
        Line2D(
            [], [],
            linestyle="",
            marker="o",
            markerfacecolor=energy_colors[ew],
            markeredgecolor="black",
            color="black",
            markersize=8,
            label=f"ew={ew:g}",
        )
        for ew in energy_weights
    ]
    fw_handles = [
        Line2D(
            [], [],
            linestyle="",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            markersize=np.sqrt(size_map[fw]),
            label=f"fw={fw:g}",
        )
        for fw in force_weights
    ]

    cut = round(cs_min + 3)
    ax_top.set_ylim(cut, cs_max*1.05)
    ax_bottom.set_ylim(cs_min*0.98, cut)

    ax_top.text(
        0.05,
        0.95,
        f"{structure}",
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        size=18,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax_bottom.text(
        0.05,
        0.05,
        "Top models",
        transform=ax_bottom.transAxes,
        va="bottom",
        ha="left",
        size=18,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )


    ax_bottom.set_xlabel(r"Cutoff $r_\mathrm{max}$ in $\AA$")
    fig.supylabel("Composite score", size=22)


    legend_seed = fig.legend(
        handles=seed_handles,
        title="Seed",
        loc="upper left",
        bbox_to_anchor=(0.9, 0.89),
    )
    fig.add_artist(legend_seed)

    legend_ew = fig.legend(
        handles=ew_handles,
        title="E-weight",
        loc="upper left",
        bbox_to_anchor=(0.9, 0.67),
    )
    fig.add_artist(legend_ew)

    fig.legend(
        handles=fw_handles,
        title="F-weight",
        loc="upper left",
        bbox_to_anchor=(0.9, 0.45),
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
    ]
    rows = require_keys(rows, keys)
    rows = sorted(rows, key=lambda r: r["score_freq_mae_term"])

    energy_weights, seeds, force_weights, size_map, energy_colors = make_style_maps(rows)

    fig, ax = plt.subplots(
        figsize=(7.5, 7.5),
        constrained_layout=False,
    )

    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]

        x = np.asarray([r["score_freq_mae_term"] for r in sub], dtype=float)
        y = np.asarray([r["score_intensity_corr_term"] for r in sub], dtype=float)

        colors = [energy_colors[float(r["energy_weight"])] for r in sub]
        sizes = [size_map[float(r["forces_weight"])] for r in sub]

        ax.scatter(
            x,
            y,
            c=colors,
            marker=marker_for_seed(seed),
            s=sizes,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.4,
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
        size_map,
        energy_colors,
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
    ]
    rows = require_keys(rows, keys)
    rows = sorted(rows, key=lambda r: r["freq_mae_ir_cm1"])

    energy_weights, seeds, force_weights, size_map, energy_colors = make_style_maps(rows)

    fig, ax = plt.subplots(
        figsize=(7.5, 7.5),
        constrained_layout=False,
    )

    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]

        x = np.asarray([r[overlap_key] for r in sub], dtype=float)
        y = np.asarray([r["freq_mae_ir_cm1"] for r in sub], dtype=float)

        colors = [energy_colors[float(r["energy_weight"])] for r in sub]
        sizes = [size_map[float(r["forces_weight"])] for r in sub]

        ax.scatter(
            x,
            y,
            c=colors,
            marker=marker_for_seed(seed),
            s=sizes,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.4,
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
        size_map,
        energy_colors,
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
    args.outdir = args.outdir / f'{args.structure}'

    if args.plot in {"all", "landscape"}:
        plot_landscape(rows, args.structure, args.outdir)

    if args.plot in {"all", "decomposition"}:
        plot_score_decomposition(rows, args.structure, args.outdir)

    if args.plot in {"all", "physics"}:
        plot_physics_quality(rows, args.structure, args.outdir)


if __name__ == "__main__":
    main()

