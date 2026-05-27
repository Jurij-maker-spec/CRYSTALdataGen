#!/usr/bin/env python3
"""
conda activate mace_env

analyse_sweep.py

Combined sweep analysis:
- global ranking diagnostics
- energy_weight-centered sensitivity plots
- hierarchical heatmap:
    x-axis: r_max groups, split by seed
    y-axis: forces_weight groups, split by energy_weight
- pareto front
- top-model peak comparison from per-run *_eval_summary.json
- frequency error histograms from per-run *_eval_summary.json
- clustering in hyperparameter space

Expected input:
    <EVAL_ROOT>/master_summary.csv
    <run_dir>/*_eval_summary.json

Outputs:
    <EVAL_ROOT>/plots/*.png
    <EVAL_ROOT>/plots/*.pdf
    <EVAL_ROOT>/csv/*.csv
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# USER SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parent

# ROOT = Path("/home/jha/jha/python_scripts/CRYSTALdataGen/results/")
# EVAL = Path("SiO2_05_95/SiO2_master_eval_full_260415_181645/")
# EVAL = Path("SiO2_25_75/SiO2_master_eval_full_260421_165802")
# EVAL = Path("SiO2_10_90/SiO2_master_eval_full_260427_170113")
# EVAL = Path("SiO2_10_90/SiO2_master_eval_full_260424_145211")
# EVAL = Path("AlN_10_90/AlN_master_eval_full_260430_092903")
# EVAL = Path("Al2O3_10_90/Al2O3_master_eval_full_260505_103502")
EVAL = Path("Al2O3_10_90/Al2O3_master_eval_full_260519_084804")


EVAL_ROOT = ROOT / 'results' / EVAL 

CSV_PATH = EVAL_ROOT / "master_summary.csv"
PLOTS_DIR = EVAL_ROOT / "plots"
CSV_DIR = EVAL_ROOT / "csv"

TOP_K = 5
N_CLUSTERS = 3

USE_MPLSTYLE = True
MPLSTYLE_PATH = "style.mplstyle"

# ============================================================
# METRIC RANKING CONFIG
# ============================================================

LOWER_IS_BETTER = [
    "composite_score",
    "freq_mae_ir_cm1",
    "freq_rmse_ir_cm1",
    "freq_mae_ir_weighted_cm1",
    "spectrum_rel_l2",
    "score_freq_mae_term",
    "score_freq_weighted_term",
    "score_spectrum_l2_term",
    "score_intensity_corr_term",
    "score_subspace_overlap_term",
    "score_mode_overlap_term",
]

HIGHER_IS_BETTER = [
    "intensity_pearson_r",
    "intensity_spearman_r",
    "mean_mode_overlap",
    "mean_subspace_overlap",
]


# ============================================================
# STYLE
# ============================================================

if USE_MPLSTYLE:
    try:
        plt.style.use(MPLSTYLE_PATH)
    except OSError:
        print(f"Warning: could not load matplotlib style: {MPLSTYLE_PATH}")


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_sorted(df: pd.DataFrame, col: str) -> list:
    vals = df[col].dropna().unique().tolist()
    return sorted(vals)


def set_discrete_ticks(ax, axis: str, values: list) -> None:
    if axis == "x":
        ax.set_xticks(values)
    elif axis == "y":
        ax.set_yticks(values)
    else:
        raise ValueError(f"Unknown axis: {axis}")


def savefig_both(fig, path_base: Path) -> None:
    """
    Save figure as PNG and PDF.

    path_base should not include a suffix.
    Example:
        savefig_both(plt.gcf(), PLOTS_DIR / "ranking")
    """
    ensure_dir(path_base.parent)
    fig.savefig(path_base.with_suffix(".png"), dpi=150)
    fig.savefig(path_base.with_suffix(".pdf"))


def find_eval_summary(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("*_eval_summary.json"))
    if not files:
        return None
    return files[0]


def short_run_label(run_name: str) -> str:
    parts = run_name.split("_")
    keep = []

    for p in parts:
        if (
            p.startswith("rmax")
            or p.startswith("seed")
            or p.startswith("ew")
            or p.startswith("fw")
        ):
            keep.append(p)

    if keep:
        return "_".join(keep)

    return run_name


def load_clean_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "eval_status" in df.columns:
        df = df[df["eval_status"] == "ok"].copy()

    required = [
        "run_name",
        "result_dir",
        "composite_score",
        "freq_mae_ir_cm1",
        "spectrum_rel_l2",
        "intensity_pearson_r",
        "r_max",
        "energy_weight",
        "seed",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in master_summary.csv: {missing}")

    if "forces_weight" not in df.columns:
        df["forces_weight"] = 0.0

    optional_metric_columns = [
        "mean_mode_overlap",
        "mean_subspace_overlap",
        "score_freq_mae_term",
        "score_freq_weighted_term",
        "score_spectrum_l2_term",
        "score_intensity_corr_term",
        "score_subspace_overlap_term",
        "score_mode_overlap_term",
    ]

    for col in optional_metric_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df.dropna(subset=["composite_score"]).copy()
    df = df.sort_values("composite_score").reset_index(drop=True)

    return df


# ============================================================
# BASIC SUMMARY
# ============================================================

def print_top_models(df: pd.DataFrame, top_k: int) -> None:
    cols = [
        "run_name",
        "composite_score",
        "freq_mae_ir_cm1",
        "spectrum_rel_l2",
        "intensity_pearson_r",
        "r_max",
        "energy_weight",
        "forces_weight",
        "seed",
    ]

    cols = [c for c in cols if c in df.columns]

    print("\nTop models:\n")
    print(df.head(top_k)[cols])


# ============================================================
# BEST-BY-METRIC SUMMARY
# ============================================================

def best_by_metric(
    df: pd.DataFrame,
    metric: str,
    higher_is_better: bool = False,
):
    if metric not in df.columns:
        return None

    sub = df[["run_name", metric]].copy()
    sub = sub.dropna(subset=[metric])

    if sub.empty:
        return None

    idx = sub[metric].idxmax() if higher_is_better else sub[metric].idxmin()

    row = df.loc[idx]

    return {
        "metric": metric,
        "direction": (
            "higher_is_better"
            if higher_is_better
            else "lower_is_better"
        ),
        "run_name": row.get("run_name"),
        "value": row.get(metric),
        "r_max": row.get("r_max"),
        "energy_weight": row.get("energy_weight"),
        "forces_weight": row.get("forces_weight"),
        "seed": row.get("seed"),
    }


def build_best_by_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in LOWER_IS_BETTER:
        rec = best_by_metric(
            df,
            metric,
            higher_is_better=False,
        )
        if rec is not None:
            rows.append(rec)

    for metric in HIGHER_IS_BETTER:
        rec = best_by_metric(
            df,
            metric,
            higher_is_better=True,
        )
        if rec is not None:
            rows.append(rec)

    return pd.DataFrame(rows)


def print_best_by_metric(best_df: pd.DataFrame) -> None:
    if len(best_df) == 0:
        print("\nNo metric ranking data available.")
        return

    print("\nBest models by metric:\n")

    cols = [
        "metric",
        "value",
        "run_name",
        "r_max",
        "energy_weight",
        "forces_weight",
        "seed",
    ]

    cols = [c for c in cols if c in best_df.columns]

    print(best_df[cols].to_string(index=False))



# ============================================================
# 1 RANKING OVERVIEW
# ============================================================

def plot_ranking(df: pd.DataFrame, plots_dir: Path) -> None:
    plt.figure()

    plt.plot(np.arange(1, len(df) + 1), df["composite_score"].values)

    plt.xlabel("ranked model")
    plt.ylabel("composite score")
    plt.title("model ranking")

    plt.tight_layout()
    savefig_both(plt.gcf(), plots_dir / "ranking")
    plt.close()


# ============================================================
# 2 FREQUENCY VS INTENSITY
# ============================================================

def plot_freq_vs_intensity(df: pd.DataFrame, plots_dir: Path) -> None:
    plt.figure()

    plt.scatter(
        df["freq_mae_ir_cm1"],
        1.0 - df["intensity_pearson_r"],
    )

    for _, row in df.head(TOP_K).iterrows():
        plt.annotate(
            short_run_label(str(row["run_name"])),
            (
                row["freq_mae_ir_cm1"],
                row["intensity_pearson_r"],
            ),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel("freq MAE IR (cm⁻¹)")
    plt.ylabel("1 - intensity correlation")
    plt.title("frequency vs intensity")

    plt.tight_layout()
    savefig_both(plt.gcf(), plots_dir / "freq_vs_intensity")
    plt.close()


# ============================================================
# 3 R_MAX SENSITIVITY
# ============================================================

def plot_rmax_sensitivity(df: pd.DataFrame, plots_dir: Path) -> None:
    r_values = unique_sorted(df, "r_max")

    plt.figure()

    for r in r_values:
        sub = df[df["r_max"] == r]

        plt.scatter(
            [r] * len(sub),
            sub["composite_score"],
            alpha=0.7,
        )

    ax = plt.gca()
    set_discrete_ticks(ax, "x", r_values)

    plt.xlabel("r_max")
    plt.ylabel("composite score")
    plt.title("cutoff sensitivity")

    plt.tight_layout()
    savefig_both(plt.gcf(), plots_dir / "rmax_sensitivity")
    plt.close()


# ============================================================
# 4 ENERGY WEIGHT SENSITIVITY
# ============================================================

def plot_energy_weight_sensitivity(df: pd.DataFrame, plots_dir: Path) -> None:
    ew_values = unique_sorted(df, "energy_weight")

    plt.figure()

    for ew in ew_values:
        sub = df[df["energy_weight"] == ew]

        plt.scatter(
            [ew] * len(sub),
            sub["composite_score"],
            alpha=0.7,
        )

    ax = plt.gca()
    set_discrete_ticks(ax, "x", ew_values)

    plt.xlabel("energy_weight")
    plt.ylabel("composite score")
    plt.title("energy weight sensitivity")

    plt.tight_layout()
    savefig_both(plt.gcf(), plots_dir / "energy_weight_sensitivity")
    plt.close()


# ============================================================
# 5 HIERARCHICAL HEATMAP
# ============================================================

def plot_hierarchical_heatmap(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Heatmap layout:

    x-axis top:
        r_max superclass

    x-axis bottom:
        seed subclass

    y-axis left:
        energy_weight subclass

    y-axis right:
        forces_weight superclass

    values:
        mean composite score
    """

    r_values = unique_sorted(df, "r_max")
    seed_values = unique_sorted(df, "seed")
    fw_values = unique_sorted(df, "forces_weight")
    ew_values = unique_sorted(df, "energy_weight")

    x_keys = [
        (r, seed)
        for r in r_values
        for seed in seed_values
    ]

    y_keys = [
        (fw, ew)
        for fw in fw_values
        for ew in ew_values
    ]

    heat = np.full((len(y_keys), len(x_keys)), np.nan)

    for iy, (fw, ew) in enumerate(y_keys):
        for ix, (r, seed) in enumerate(x_keys):
            sub = df[
                (df["forces_weight"] == fw)
                & (df["energy_weight"] == ew)
                & (df["r_max"] == r)
                & (df["seed"] == seed)
            ]

            if len(sub) > 0:
                heat[iy, ix] = sub["composite_score"].mean()

    fig_width = max(10, 0.7 * len(x_keys))
    fig_height = max(5, 0.45 * len(y_keys))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(heat, aspect="auto")

    # Bottom x-axis: seed labels
    ax.set_xticks(range(len(x_keys)))
    ax.set_xticklabels([f"s{seed:g}" for _, seed in x_keys])
    ax.set_xlabel("seed within each r_max")

    # Left y-axis: energy_weight labels
    ax.set_yticks(range(len(y_keys)))
    ax.set_yticklabels([f"{ew:g}" for _, ew in y_keys])
    ax.set_ylabel("energy_weight")

    # Right y-axis: forces_weight group labels
    ax_right = ax.secondary_yaxis("right")

    fw_group_centers = []
    for i, _fw in enumerate(fw_values):
        start = i * len(ew_values)
        end = start + len(ew_values) - 1
        fw_group_centers.append((start + end) / 2)

    ax_right.set_yticks(fw_group_centers)
    ax_right.set_yticklabels([f"{fw:g}" for fw in fw_values])
    ax_right.set_ylabel("forces_weight")

    # Top x-axis: r_max group labels
    ax_top = ax.secondary_xaxis("top")

    r_group_centers = []
    for i, _r in enumerate(r_values):
        start = i * len(seed_values)
        end = start + len(seed_values) - 1
        r_group_centers.append((start + end) / 2)

    ax_top.set_xticks(r_group_centers)
    ax_top.set_xticklabels([f"r_max={r:g}" for r in r_values])
    ax_top.set_xlabel("r_max")

    # Vertical separators between r_max groups
    for i in range(1, len(r_values)):
        ax.axvline(
            i * len(seed_values) - 0.5,
            color="k",
            linewidth=0.8,
        )

    # Horizontal separators between forces_weight groups
    for i in range(1, len(fw_values)):
        ax.axhline(
            i * len(ew_values) - 0.5,
            color="k",
            linewidth=0.8,
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean composite score")

    ax.set_title(
        "Mean composite score by forces_weight, energy_weight, r_max, and seed"
    )

    plt.tight_layout()
    savefig_both(fig, plots_dir / "heatmap_fw_ew_rmax_seed")
    plt.close(fig)


# ============================================================
# 6 PARETO FRONT
# ============================================================

def plot_pareto(df: pd.DataFrame, plots_dir: Path) -> None:
    freq = df["freq_mae_ir_cm1"].values
    spec = df["spectrum_rel_l2"].values

    pareto = []
    pareto_points = []

    for i in range(len(df)):
        dominated = False

        for j in range(len(df)):
            if (
                freq[j] <= freq[i]
                and spec[j] <= spec[i]
                and (
                    freq[j] < freq[i]
                    or spec[j] < spec[i]
                )
            ):
                dominated = True
                break

        if not dominated:
            pareto.append(i)
            pareto_points.append((freq[i], spec[i], i))
            
    pareto_points = sorted(pareto_points, key=lambda x: x[0])

    if len(pareto_points) > 1:
        plt.plot(
            [p[0] for p in pareto_points],
            [p[1] for p in pareto_points],
            linewidth=1.0,
        )

    plt.figure()

    plt.scatter(freq, spec, alpha=0.7)

    plt.scatter(
        freq[pareto],
        spec[pareto],
        marker="x",
        s=80,
    )
    for i in pareto:
        row = df.iloc[i]

        plt.annotate(
            short_run_label(str(row["run_name"])),
            (freq[i], spec[i]),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel("freq MAE (cm⁻¹)")
    plt.ylabel("spectrum rel L2")
    plt.title("pareto front")

    plt.tight_layout()
    savefig_both(plt.gcf(), plots_dir / "pareto")
    plt.close()


# ============================================================
# 7 PEAK POSITION COMPARISON
# ============================================================

def extract_gamma_mode_matches(summary: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract full Gamma phonon mode comparison from eval summary.

    Expected source:
        summary["crystal_mode_comparison"]["matches"]

    Returns
    -------
    ref_freqs : ndarray
    pred_freqs : ndarray
    overlaps : ndarray
    """
    cmp = summary.get("crystal_mode_comparison", {})
    matches = cmp.get("matches", [])

    ref_freqs = []
    pred_freqs = []
    overlaps = []

    for m in matches:
        if not isinstance(m, dict):
            continue

        # current mode_matching.py keys
        ref = m.get("freq_ref", m.get("ref_freq", None))
        pred = m.get("freq_test", m.get("test_freq", None))
        ov = m.get("overlap", np.nan)

        if ref is None or pred is None:
            continue

        ref_freqs.append(float(ref))
        pred_freqs.append(float(pred))
        overlaps.append(float(ov))

    return (
        np.asarray(ref_freqs, dtype=float),
        np.asarray(pred_freqs, dtype=float),
        np.asarray(overlaps, dtype=float),
    )


def plot_gamma_phonon_comparison(df: pd.DataFrame, plots_dir: Path, top_k: int) -> None:
    """
    Full Gamma phonon frequency comparison:
        CRYSTAL Hessian/ref_db modes vs MACE analytical Hessian modes.
    """
    fig, ax = plt.subplots()

    max_freq_seen = 0.0
    plotted_any = False

    for _, row in df.head(top_k).iterrows():
        run_dir = Path(row["result_dir"])
        summary_file = find_eval_summary(run_dir)

        if summary_file is None:
            print(f"Warning: no eval summary found in {run_dir}")
            continue

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        ref, pred, overlaps = extract_gamma_mode_matches(summary)

        if len(ref) == 0 or len(pred) == 0:
            print(f"Warning: no Gamma mode matches in {summary_file}")
            continue

        max_freq_seen = max(
            max_freq_seen,
            float(np.nanmax(ref)),
            float(np.nanmax(pred)),
        )

        sc = ax.scatter(
            ref,
            pred,
            c=overlaps,
            vmin=0.0,
            vmax=1.0,
            alpha=0.75,
            label=short_run_label(str(row["run_name"])),
        )

        plotted_any = True

    if plotted_any:
        x = np.linspace(0.0, max_freq_seen * 1.05, 200)
        ax.plot(x, x, ls="--", color="black", linewidth=1.0)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("mode overlap")

    ax.set_xlabel("CRYSTAL Γ frequency (cm⁻¹)")
    ax.set_ylabel("MACE Γ frequency (cm⁻¹)")
    # ax.set_title("Γ phonon frequency comparison")

    if plotted_any:
        ax.legend(fontsize=6)

    fig.tight_layout()
    savefig_both(fig, plots_dir / "gamma_phonon_frequency_comparison")
    plt.close(fig)


# ============================================================
# 8 FREQUENCY ERROR HISTOGRAM
# ============================================================

def plot_gamma_frequency_error_hist(df: pd.DataFrame, plots_dir: Path, top_k: int) -> None:
    """
    Histogram of full Gamma phonon frequency errors:
        error = MACE - CRYSTAL
    """
    fig, ax = plt.subplots()

    plotted_any = False

    for _, row in df.head(top_k).iterrows():
        run_dir = Path(row["result_dir"])
        summary_file = find_eval_summary(run_dir)

        if summary_file is None:
            print(f"Warning: no eval summary found in {run_dir}")
            continue

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        ref, pred, overlaps = extract_gamma_mode_matches(summary)

        if len(ref) == 0 or len(pred) == 0:
            print(f"Warning: no Gamma mode matches in {summary_file}")
            continue

        err = pred - ref

        ax.hist(
            err,
            bins=20,
            alpha=0.45,
            label=short_run_label(str(row["run_name"])),
        )

        plotted_any = True

    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Γ phonon frequency error: MACE - CRYSTAL (cm⁻¹)")
    ax.set_ylabel("count")
    ax.set_title("Γ phonon frequency error distribution")

    if plotted_any:
        ax.legend(fontsize=6)

    fig.tight_layout()
    savefig_both(fig, plots_dir / "gamma_frequency_error_hist")
    plt.close(fig)



# ============================================================
# 9 CLUSTERING
# ============================================================

def run_clustering(df: pd.DataFrame, csv_dir: Path, n_clusters: int) -> pd.DataFrame:
    feature_cols = [
        "r_max",
        "energy_weight",
        "seed",
        "batch_size",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    if len(df) < n_clusters:
        print(
            f"Skipping clustering: n_runs={len(df)} smaller than "
            f"n_clusters={n_clusters}"
        )
        df = df.copy()
        df["cluster"] = -1
        return df

    features = df[feature_cols].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=0,
    )

    labels = kmeans.fit_predict(X)

    df = df.copy()
    df["cluster"] = labels

    cluster_stats = df.groupby("cluster")[
        [
            "composite_score",
            "freq_mae_ir_cm1",
            "spectrum_rel_l2",
            "intensity_pearson_r",
        ]
    ].mean()

    cluster_stats.to_csv(csv_dir / "cluster_stats.csv")

    print("\nCluster stats:\n")
    print(cluster_stats)

    robust_cols = [
        "r_max",
        "energy_weight",
        "forces_weight",
        "batch_size",
        "seed",
    ]
    robust_cols = [c for c in robust_cols if c in df.columns]

    best_cluster = cluster_stats["composite_score"].idxmin()
    robust_region = df[df["cluster"] == best_cluster][robust_cols]

    robust_region.describe().to_csv(csv_dir / "robust_region_summary.csv")

    print("\nBest cluster:", best_cluster)
    print("\nRobust parameter region:")
    print(robust_region.describe())

    return df


def plot_cluster_map(df: pd.DataFrame, plots_dir: Path) -> None:
    if "cluster" not in df.columns:
        return

    r_values = unique_sorted(df, "r_max")
    ew_values = unique_sorted(df, "energy_weight")

    plt.figure()

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]

        plt.scatter(
            sub["r_max"],
            sub["energy_weight"],
            label=f"cluster {c}",
            alpha=0.7,
        )

    ax = plt.gca()
    set_discrete_ticks(ax, "x", r_values)
    set_discrete_ticks(ax, "y", ew_values)

    plt.xlabel("r_max")
    plt.ylabel("energy_weight")
    plt.title("hyperparameter clusters")
    plt.legend()

    plt.tight_layout()
    savefig_both(plt.gcf(), plots_dir / "cluster_map")
    plt.close()


def plot_cluster_quality(df: pd.DataFrame, plots_dir: Path) -> None:
    if "cluster" not in df.columns:
        return

    clusters = sorted(df["cluster"].unique())

    plt.figure()

    for c in clusters:
        sub = df[df["cluster"] == c]

        plt.scatter(
            [c] * len(sub),
            sub["composite_score"],
            alpha=0.7,
        )

    ax = plt.gca()
    set_discrete_ticks(ax, "x", clusters)

    plt.xlabel("cluster")
    plt.ylabel("composite score")
    plt.title("cluster performance")

    plt.tight_layout()
    savefig_both(plt.gcf(), plots_dir / "cluster_quality")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dir(PLOTS_DIR)
    ensure_dir(CSV_DIR)

    df = load_clean_dataframe(CSV_PATH)

    print_top_models(df, TOP_K)
    best_metric_df = build_best_by_metric_summary(df)

    print_best_by_metric(best_metric_df)

    best_metric_df.to_csv(
        CSV_DIR / "best_by_metric.csv",
        index=False,
    )

    df.head(TOP_K).to_csv(CSV_DIR / f"top{TOP_K}.csv", index=False)

    plot_ranking(df, PLOTS_DIR)
    plot_freq_vs_intensity(df, PLOTS_DIR)
    plot_rmax_sensitivity(df, PLOTS_DIR)
    plot_energy_weight_sensitivity(df, PLOTS_DIR)
    plot_hierarchical_heatmap(df, PLOTS_DIR)
    plot_pareto(df, PLOTS_DIR)

    plot_gamma_phonon_comparison(df, PLOTS_DIR, TOP_K)
    plot_gamma_frequency_error_hist(df, PLOTS_DIR, TOP_K)

    # df_clustered = run_clustering(df, CSV_DIR, N_CLUSTERS)
    # plot_cluster_map(df_clustered, PLOTS_DIR)
    # plot_cluster_quality(df_clustered, PLOTS_DIR)
    # df_clustered.to_csv(CSV_DIR / "analysis_with_clusters.csv", index=False)
    df.to_csv(CSV_DIR / "analysis_clean.csv", index=False)

    print("\nAnalysis saved to:")
    print(f"  plots: {PLOTS_DIR}")
    print(f"  csv  : {CSV_DIR}")


if __name__ == "__main__":
    main()
