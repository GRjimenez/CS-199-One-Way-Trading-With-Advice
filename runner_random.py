"""
runner_random.py
================
Monte Carlo analyzer for the Randomized EXPO algorithm (El-Yaniv et al. 2001).

Runs 1,000 trials per slice per k-bit level, exposing the true distribution
of competitive ratios — mean, median, best case, worst case, and variance.

Key insight from El-Yaniv Theorem 1:
    Randomized one-way trading = deterministic one-way trading in expectation.
    So what we're really measuring here is the VARIANCE of the randomized
    search algorithm — how much the result swings depending on which
    reservation price is randomly chosen.

    High variance = risky (sometimes great, sometimes terrible)
    Low variance  = consistent (k-bit advice should tighten this)

Outputs per slice:
    - Statistical summary table (mean, median, best, worst, std dev, % best)
    - Distribution histogram showing the full CR spread per k-bit level
    - Convergence chart showing how mean CR stabilises over trials
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from randomized_expo import RandomizedExpoTrader

REPORT_DIR = "reports_random"
os.makedirs(REPORT_DIR, exist_ok=True)

K_BITS_LIST = [0, 1, 2, 3]

COLOURS = {
    0: "#6c757d",   # grey
    1: "#2196F3",   # blue
    2: "#FF9800",   # orange
    3: "#4CAF50",   # green
}

LABELS = {
    0: "EXPO 0-bit (no advice)",
    1: "EXPO 1-bit advice",
    2: "EXPO 2-bit advice",
    3: "EXPO 3-bit advice",
}


# =============================================================================
# Helpers
# =============================================================================

def oracle_advice_index(true_max, global_m, global_M, k_bits):
    num_intervals = 2 ** k_bits
    ratio = global_M / global_m
    for i in range(num_intervals):
        upper = global_m * (ratio ** ((i + 1) / num_intervals))
        if true_max <= upper or i == num_intervals - 1:
            return i
    return num_intervals - 1


def prepare_df(file_path, price_col):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: file not found — {file_path}")
        return None

    df = df.iloc[::-1].reset_index(drop=True)
    for c in ["Low", "High", "Open", "Close/Last", "Price"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r"[\$,]", "", regex=True)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [col for col in ["Low", "High", price_col] if col in df.columns]
    df = df.dropna(subset=keep)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# =============================================================================
# Monte Carlo Engine
# =============================================================================

def run_monte_carlo(prices, m, M, n, k_bits, runs=1000, seed=42):
    """
    Run RandomizedExpoTrader `runs` times and collect all cash outcomes.
    Uses a seeded numpy rng for full reproducibility.
    """
    true_max = float(np.max(prices))
    idx = oracle_advice_index(true_max, m, M, k_bits) if k_bits > 0 else 0

    # Single rng passed to every trader — reproducible Monte Carlo
    rng = np.random.default_rng(seed)

    all_cashes = np.empty(runs)
    theoretical_c = None

    for i in range(runs):
        trader = RandomizedExpoTrader(
            m=m, M=M, n=n, k_bits=k_bits, advice_index=idx, rng=rng
        )
        if theoretical_c is None:
            theoretical_c = trader.c

        for day, p in enumerate(prices):
            trader.trade(float(p), day + 1, "")

        all_cashes[i] = trader.cash

    return all_cashes, theoretical_c


# =============================================================================
# Chart 1: Distribution Histogram
# =============================================================================

def plot_distribution(cr_by_k, stats_by_k, label, filename):
    """
    Side-by-side histogram of CR distributions for each k-bit level.
    Shows mean (dashed) and median (solid) lines.
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#f8f9fa")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    for idx, k in enumerate(K_BITS_LIST):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#ffffff")

        crs = cr_by_k[k]
        st = stats_by_k[k]

        # Histogram
        weights = np.ones_like(crs) / len(crs)
        n_bins = min(50, max(10, len(np.unique(np.round(crs, 3)))))
        ax.hist(crs, bins=n_bins, weights=weights,
                color=COLOURS[k], alpha=0.7,
                edgecolor="white", linewidth=0.5)

        # Mean line
        ax.axvline(st["mean"], color=COLOURS[k], linestyle="--",
                   linewidth=2.0, label=f"Mean = {st['mean']:.3f}")
        # Median line
        ax.axvline(st["median"], color="#1a1a2e", linestyle="-",
                   linewidth=1.5, alpha=0.7, label=f"Median = {st['median']:.3f}")
        # OPT line (CR=1)
        ax.axvline(1.0, color="#E91E63", linestyle=":",
                   linewidth=1.5, alpha=0.8, label="OPT (CR=1)")

        # Shade best-case region
        ax.axvspan(crs.min(), st["mean"], alpha=0.08, color=COLOURS[k])

        ax.set_title(f"{LABELS[k]}\nc = {st['c']:.4f}  |  σ = {st['std']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Competitive Ratio (lower = better)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--")

        # Annotation box with key stats
        textstr = (f"Best:  {st['best']:.3f}\n"
                   f"Worst: {st['worst']:.3f}\n"
                   f"% ≤ Mean: {st['pct_below_mean']:.1f}%")
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=COLOURS[k], alpha=0.9))

    fig.suptitle(f"Randomized EXPO — CR Distribution (1,000 runs)\n{label}",
                 fontsize=13, fontweight="bold", y=1.01)

    path = os.path.join(REPORT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    return path


# =============================================================================
# Chart 2: Convergence of Mean CR over Trials
# =============================================================================

def plot_convergence(cr_by_k, label, filename):
    """
    Shows how the running mean of CR stabilises as number of trials increases.
    Validates that 1,000 runs is sufficient.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    for k in K_BITS_LIST:
        crs = cr_by_k[k]
        running_mean = np.cumsum(crs) / np.arange(1, len(crs) + 1)
        ax.plot(np.arange(1, len(crs) + 1), running_mean,
                color=COLOURS[k], linewidth=1.8, label=LABELS[k])

        # Final mean annotation
        ax.annotate(f"{running_mean[-1]:.3f}",
                    xy=(len(crs), running_mean[-1]),
                    xytext=(len(crs) - 80, running_mean[-1]),
                    fontsize=8, color=COLOURS[k], fontweight="bold")

    ax.set_xlabel("Number of Trials", fontsize=11)
    ax.set_ylabel("Running Mean Competitive Ratio", fontsize=11)
    ax.set_title(f"Convergence of Mean CR over Monte Carlo Trials\n{label}",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(1, len(cr_by_k[0]))

    path = os.path.join(REPORT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    return path


# =============================================================================
# Chart 3: Summary Comparison Bar Chart
# =============================================================================

def plot_summary_bars(stats_by_k, label, filename):
    """
    Grouped bar chart: mean CR per k-bit with error bars showing std dev.
    Makes it easy to see both the improvement and the variance reduction.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    ks = K_BITS_LIST
    means = [stats_by_k[k]["mean"] for k in ks]
    stds = [stats_by_k[k]["std"] for k in ks]
    bests = [stats_by_k[k]["best"] for k in ks]
    worsts = [stats_by_k[k]["worst"] for k in ks]

    x = np.arange(len(ks))
    bars = ax.bar(x, means, color=[COLOURS[k] for k in ks],
                  alpha=0.8, edgecolor="white", linewidth=1.5, width=0.5,
                  yerr=stds, capsize=6, error_kw={"linewidth": 2, "color": "#333"})

    # Best/worst range markers
    for i, k in enumerate(ks):
        ax.plot([x[i], x[i]], [bests[i], worsts[i]],
                color="#1a1a2e", linewidth=1, alpha=0.4)
        ax.scatter([x[i]], [bests[i]], marker="^", color=COLOURS[k],
                   s=60, zorder=5, label="_nolegend_")
        ax.scatter([x[i]], [worsts[i]], marker="v", color=COLOURS[k],
                   s=60, zorder=5, label="_nolegend_")

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # OPT reference line
    ax.axhline(1.0, color="#E91E63", linestyle="--",
               linewidth=1.5, alpha=0.8, label="OPT (CR = 1.0)")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in ks], fontsize=9)
    ax.set_ylabel("Mean Competitive Ratio ± Std Dev", fontsize=11)
    ax.set_title(f"Mean CR with Variance by Advice Level\n{label}",
                 fontsize=12, fontweight="bold", pad=12)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color="#E91E63", linestyle="--", label="OPT (CR=1.0)"),
        Line2D([0], [0], marker="^", color="#333", linestyle="None",
               markersize=8, label="Best run"),
        Line2D([0], [0], marker="v", color="#333", linestyle="None",
               markersize=8, label="Worst run"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax.set_ylim(bottom=0.9)

    path = os.path.join(REPORT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    return path


# =============================================================================
# Main simulation function
# =============================================================================

def run_detailed_simulation(df, price_col, label, slug, runs=1000):
    if df is None or len(df) == 0:
        return

    prices = df[price_col].values
    m = float(np.min(prices))
    M = float(np.max(prices))
    n = len(prices)
    opt_cash = M * 100.0

    print(f"\n{'═'*90}")
    print(f"  {label}")
    print(f"  Monte Carlo Report  |  {runs:,} trials per k-bit level")
    print(f"{'═'*90}")
    print(f"  n = {n} days  |  m = ${m:.2f}  |  M = ${M:.2f}  "
          f"|  M/m = {M/m:.3f}  |  OPT = ${opt_cash:,.2f}")
    print()

    # Column headers
    print(f"  {'Advice':>16} │ {'c':>6} │ {'Mean CR':>8} │ {'Median':>8} │ "
          f"{'Best':>7} │ {'Worst':>7} │ {'Std Dev':>7} │ {'% ≤ Mean':>8} │ {'vs 0-bit':>9}")
    print("  " + "─" * 88)

    cr_by_k = {}
    stats_by_k = {}
    base_mean = None

    for k in K_BITS_LIST:
        cashes, theor_c = run_monte_carlo(prices, m, M, n, k, runs=runs)
        opt_cash_arr = np.full_like(cashes, opt_cash)
        crs = np.divide(opt_cash_arr, cashes,
                        out=np.full_like(cashes, np.inf),
                        where=cashes > 1e-9)

        st = {
            "c":              theor_c,
            "mean":           float(np.mean(crs)),
            "median":         float(np.median(crs)),
            "best":           float(np.min(crs)),
            "worst":          float(np.max(crs)),
            "std":            float(np.std(crs)),
            "pct_below_mean": float(np.mean(crs <= np.mean(crs)) * 100),
        }
        cr_by_k[k] = crs
        stats_by_k[k] = st

        if k == 0:
            base_mean = st["mean"]
            vs_base = "—"
        else:
            if base_mean and math.isfinite(base_mean) and math.isfinite(st["mean"]):
                pct = (base_mean - st["mean"]) / base_mean * 100
                vs_base = f"{pct:+.2f}%"
            else:
                vs_base = "N/A"

        print(f"  {LABELS[k]:>16} │ {theor_c:>6.4f} │ {st['mean']:>8.4f} │ "
              f"{st['median']:>8.4f} │ {st['best']:>7.4f} │ {st['worst']:>7.4f} │ "
              f"{st['std']:>7.4f} │ {st['pct_below_mean']:>7.1f}% │ {vs_base:>9}")

    print()

    # Generate all three charts
    dist_path  = plot_distribution(cr_by_k, stats_by_k, label, f"dist_{slug}.png")
    conv_path  = plot_convergence(cr_by_k, label, f"conv_{slug}.png")
    bars_path  = plot_summary_bars(stats_by_k, label, f"bars_{slug}.png")

    print(f"  Charts saved:")
    print(f"    Distribution  → {dist_path}")
    print(f"    Convergence   → {conv_path}")
    print(f"    Summary bars  → {bars_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    FILE_PATH = "HistoricalData_1773022846406.csv"
    PRICE_COL = "Close/Last"
    NAME      = "Apple Stock"
    RUNS      = 1000

    print(__doc__)

    df = prepare_df(FILE_PATH, PRICE_COL)
    if df is None:
        raise SystemExit(1)

    # Day-based slices
    for n_days in [10, 30, 60, 90, 180]:
        if len(df) < n_days:
            continue
        run_detailed_simulation(
            df.iloc[:n_days].reset_index(drop=True),
            PRICE_COL,
            f"{NAME} — First {n_days} Days",
            f"apple_{n_days}d",
            runs=RUNS,
        )

    # Yearly slices
    if "Date" in df.columns and df["Date"].notna().any():
        years = sorted(df["Date"].dt.year.dropna().unique())
        for k_yr in range(1, min(4, len(years) + 1)):
            yr_slice = years[:k_yr]
            df_yr = df[df["Date"].dt.year.isin(yr_slice)].reset_index(drop=True)
            run_detailed_simulation(
                df_yr,
                PRICE_COL,
                f"{NAME} — {k_yr} Year(s) ({yr_slice[0]}–{yr_slice[-1]})",
                f"apple_{k_yr}yr",
                runs=RUNS,
            )

    print(f"\n{'═'*90}")
    print(f"  All Monte Carlo analysis complete. Reports saved to ./{REPORT_DIR}/")
    print(f"{'═'*90}")