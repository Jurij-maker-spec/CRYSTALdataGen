#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from util.ref_db import read_crystal_ir_reference
from util.plotting import restore_degeneracies, perform_KDE

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REF_DB = PROJECT_ROOT / "data" / "ref_db.h5"
DEFAULT_STYLE = PROJECT_ROOT / "util" / "style.mplstyle"
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "summary_plots"

STRUCTURE_PAIRS = [
    ("SiO2", "SiO2_PBE", r"SiO$_2$"),
    ("AlN", "AlN_PBE", "AlN"),
    ("Al2O3", "Al2O3_PBE", r"Al$_2$O$_3$"),
    ("TiO2_rutil", "TiO2_rutil_PBE", r"TiO$_2$ rutile"),
]
CMAP = mpl.colormaps["viridis"]


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


def plot_hse_pbe_ir_grid(
    *,
    ref_db: Path,
    outdir: Path,
    outfile_stem: str,
    fwhm: float,
    style_path: Path | None,
) -> None:
    apply_style(style_path)

    colors = CMAP([0.35, 0.75])

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

        ax.plot(x_hse, y_hse, lw=1.5, color=colors[0], label="HSEsol" if i == 0 else '')
        ax.plot(x_pbe, y_pbe, lw=1.5, ls='--', color=colors[1], label="PBEsol" if i == 0 else '')
        ax.text(
            0.50,
            0.95,
            title,
            transform=ax.transAxes,
            va='top', 
            ha='center',
            size=18,
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.05), 
            )
        i += 1
    
    fig.supylabel('Relative IR intensity')
    fig.supxlabel(r"Wavenumber in cm$^{-1}$")
    fig.legend(bbox_to_anchor=(1.15, 0.62))

    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / f"{outfile_stem}.png"
    pdf = outdir / f"{outfile_stem}.pdf"

    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


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
        "hse-pbe-ir-grid",
        help="Plot HSEsol vs PBEsol CRYSTAL reference IR spectra in a 2x2 grid.",
    )
    add_common_args(p_grid)
    p_grid.add_argument("--outfile-stem", default="hse_pbe_reference_ir_grid")
    p_grid.add_argument("--fwhm", type=float, default=12.0)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "hse-pbe-ir-grid":
        plot_hse_pbe_ir_grid(
            ref_db=args.ref_db,
            outdir=args.outdir,
            outfile_stem=args.outfile_stem,
            fwhm=args.fwhm,
            style_path=args.style,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

