"""
runner_restricted.py
====================
Oracle Robustness Test — how little data does the oracle need to beat
the no-advice baseline?

Setup:
    - Full trading period = 5 years (all available data up to 5yr mark)
    - Baseline = threat-based 0-bit trades the full 5 years (flat line)
    - k-bit advice = oracle only sees the first X months to pick its
      sub-interval, then algorithm trades the full 5 years with that advice

    X sweeps from 1 to 60 months (monthly resolution).

    Key question: at which month does partial advice first beat the baseline?

Two separate charts generated:
    1. Threat-Based Oracle Robustness
    2. Randomized EXPO Oracle Robustness

Datasets:
    - Apple Stock (HistoricalData_1773022846406.csv)
    - USD → PHP   (USD_PHP.csv)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from threat_based import BaseThreatTrader
from k_bit_advice import KBitThreatTrader
from randomized_expo import RandomizedExpoTrader

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

EXPO_RUNS = 1000   # Monte Carlo runs per month/k combo for EXPO
EXPO_SEED = 42

COLOURS = {
    "baseline": "#E91E63",
    1: "#2196F3",
    2: "#FF9800",
    3: "#4CAF50",
    4: "#9C27B0",
    8: "#8BC34A",
}


# =============================================================================
# Helpers
# =============================================================================

def oracle_advice_index(restricted_max, global_m, global_M, k_bits):
    """
    Returns the sub-interval index based ONLY on the restricted window max.
    When restricted_max < true_max the advice will be wrong (too low),
    which is exactly what this experiment tests.
    """
    num_intervals = 2 ** k_bits
    ratio = global_M / global_m
    for i in range(num_intervals):
        upper = global_m * (ratio ** ((i + 1) / num_intervals))
        if restricted_max <= upper or i == num_intervals - 1:
            return i
    return num_intervals - 1


def prepare_df(file_path, price_col):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"  [!] File not found: {file_path}")
        return None

    df = df.iloc[::-1].reset_index(drop=True)   # oldest first

    for c in ["Low", "High", "Open", "Close/Last", "Price", "Close"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r"[\$,]", "", regex=True)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[price_col])

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def get_5yr_slice(df, price_col):
    """
    Return the first 5 years of data using dates if available,
    otherwise fall back to first 1302 rows (~5yr of trading days).
    """
    if "Date" in df.columns and df["Date"].notna().any():
        start = df["Date"].dropna().iloc[0]
        end   = start + pd.DateOffset(years=5)
        df_5yr = df[df["Date"] < end].reset_index(drop=True)
        if len(df_5yr) > 50:
            return df_5yr

    # Fallback: ~252 trading days/yr × 5 = 1260; use 1302 to be safe
    return df.head(min(1302, len(df))).reset_index(drop=True)


def month_window_size(month, total_days, total_months=60):
    """
    Convert a month index (1–60) to a day count into the dataset.
    Rounds up so month 1 always has at least 1 day.
    """
    days_per_month = total_days / total_months
    return min(int(math.ceil(month * days_per_month)), total_days)


# =============================================================================
# Plotting
# =============================================================================

def plot_robustness_chart(x_months, baseline_cr, results_dict,
                          k_bits_list, title, filename,
                          annotate_drops=False):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    # ── OPT reference ────────────────────────────────────────────────────────
    ax.axhline(1.0, color="#1a1a2e", linewidth=2.0,
               linestyle=":", alpha=0.7, label="Optimal (CR = 1.0)", zorder=2)

    # ── Shaded region where advice BEATS baseline ─────────────────────────────
    # Shade BELOW the baseline (where CR is lower = better)
    ax.axhspan(1.0, baseline_cr, facecolor="#4CAF50",
               alpha=0.07, label="_nolegend_", zorder=1)

    # ── Baseline flat line ────────────────────────────────────────────────────
    ax.axhline(baseline_cr, color=COLOURS["baseline"], linewidth=2.5,
               linestyle="--", zorder=3,
               label=f"Baseline 0-Bit (CR: {baseline_cr:.3f})")

    # ── k-bit advice lines ────────────────────────────────────────────────────
    all_vals = [baseline_cr, 1.0]

    for k in k_bits_list:
        y_vals = results_dict[k]
        colour = COLOURS.get(k, "#000000")

        ax.plot(x_months, y_vals, color=colour, linewidth=2.2,
                label=f"{k}-Bit Advice", alpha=0.92, zorder=4)

        finite_vals = [v for v in y_vals if math.isfinite(v)]
        all_vals.extend(finite_vals)

        # Annotate drop points (only for threat-based step functions)
        if annotate_drops:
            for i in range(1, len(y_vals)):
                if math.isfinite(y_vals[i-1]) and math.isfinite(y_vals[i]):
                    if y_vals[i-1] - y_vals[i] > 0.05:   # significant drop
                        ax.scatter(x_months[i], y_vals[i], color=colour,
                                   s=70, marker="v", zorder=5)

    # ── Axis labels and formatting ────────────────────────────────────────────
    ax.set_xlabel("Oracle Data Fed (Months)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Final Expected Competitive Ratio (Full 5 Years)",
                  fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.legend(loc="upper right", fontsize=11, framealpha=0.95,
              edgecolor="#cccccc", borderpad=0.8)
    ax.grid(True, alpha=0.3, linestyle="--")

    ax.set_xticks(np.arange(0, 61, 6))
    ax.set_xlim(1, 60)

    # Dynamic y-axis: show from just below OPT to just above the max value
    finite_all = [v for v in all_vals if math.isfinite(v)]
    y_top = max(finite_all) * 1.06 if finite_all else baseline_cr * 1.1
    y_bot = max(0.97, min(finite_all) * 0.97) if finite_all else 0.97
    ax.set_ylim(bottom=y_bot, top=y_top)

    plt.tight_layout(pad=2.0)
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print(f"    → Saved: {path}")


# =============================================================================
# Simulation core
# =============================================================================

def run_restricted_simulation(df, price_col, k_bits_list, name, prefix):
    df_5yr  = get_5yr_slice(df, price_col)
    prices  = df_5yr[price_col].values
    n       = len(prices)
    global_m = float(np.min(prices))
    global_M = float(np.max(prices))
    opt_cash = global_M * 100.0
    total_months = 60

    print(f"\n{'='*75}")
    print(f"  ORACLE ROBUSTNESS: {name}")
    print(f"{'='*75}")
    print(f"  Days={n}  |  m=${global_m:.2f}  |  M=${global_M:.2f}  |  M/m={global_M/global_m:.3f}")

    # ── Baselines (0-bit, no advice) ──────────────────────────────────────────
    try:
        tb = BaseThreatTrader(n=n, m=global_m, M=global_M)
        for i, p in enumerate(prices):
            tb.trade(float(p), i + 1, "")
        threat_baseline_cr = opt_cash / tb.cash if tb.cash > 1e-9 else float("inf")
    except Exception as e:
        print(f"  [!] Threat baseline failed: {e}")
        threat_baseline_cr = float("inf")

    rng = np.random.default_rng(EXPO_SEED)
    expo_cashes = []
    for _ in range(EXPO_RUNS):
        try:
            et = RandomizedExpoTrader(m=global_m, M=global_M, n=n, k_bits=0, rng=rng)
            for i, p in enumerate(prices):
                et.trade(float(p), i + 1, "")
            expo_cashes.append(et.cash)
        except Exception:
            pass
    expo_baseline_cr = (opt_cash / np.mean(expo_cashes)
                        if expo_cashes else float("inf"))

    print(f"  Threat 0-bit baseline CR : {threat_baseline_cr:.4f}")
    print(f"  EXPO  0-bit baseline CR  : {expo_baseline_cr:.4f}")
    print(f"  Running {total_months} monthly windows × {len(k_bits_list)} k-bit levels...")

    # ── Monthly sweep ─────────────────────────────────────────────────────────
    x_months      = np.arange(1, total_months + 1)
    threat_results = {k: [] for k in k_bits_list}
    expo_results   = {k: [] for k in k_bits_list}

    for month in x_months:
        window = month_window_size(month, n, total_months)
        restricted_max = float(np.max(prices[:window]))

        for k in k_bits_list:
            idx = oracle_advice_index(restricted_max, global_m, global_M, k)

            # -- Threat-based --
            try:
                tt = KBitThreatTrader(n=n, m=global_m, M=global_M,
                                      k_bits=k, advice_index=idx)
                for i, p in enumerate(prices):
                    tt.trade(float(p), i + 1, "")
                t_cr = opt_cash / tt.cash if tt.cash > 1e-9 else float("inf")
            except Exception:
                t_cr = threat_baseline_cr   # fallback to baseline on error
            threat_results[k].append(t_cr)

            # -- Randomized EXPO (Monte Carlo average) --
            rng_k = np.random.default_rng(EXPO_SEED + month * 100 + k)
            e_cashes = []
            for _ in range(EXPO_RUNS):
                try:
                    et = RandomizedExpoTrader(m=global_m, M=global_M, n=n,
                                             k_bits=k, advice_index=idx,
                                             rng=rng_k)
                    for i, p in enumerate(prices):
                        et.trade(float(p), i + 1, "")
                    e_cashes.append(et.cash)
                except Exception:
                    pass
            e_cr = (opt_cash / np.mean(e_cashes)
                    if e_cashes else float("inf"))
            expo_results[k].append(e_cr)

        if month % 10 == 0:
            print(f"    Month {month:2d}/60 done")

    # ── Crossover Report ─────────────────────────────────────────────────────
    # For each algorithm type and each k-bit level, find:
    #   1. The FIRST month where CR drops below the baseline (crossover point)
    #   2. Whether it stays below baseline consistently after that
    #   3. The final CR at month 60 and improvement over baseline

    def print_crossover_report(results, baseline_cr, algo_label):
        print(f"\n  {'─'*70}")
        print(f"  {algo_label} — Crossover Report  (baseline CR = {baseline_cr:.4f})")
        print(f"  {'─'*70}")
        print(f"  {'k-bits':>8} │ {'First beats':>12} │ {'Stays below':>12} │ "
              f"{'Final CR':>9} │ {'Improvement':>12} │ {'Min CR':>8}")
        print(f"  {'─'*70}")

        for k in k_bits_list:
            y_vals = results[k]

            # First month where CR < baseline
            first_beat = None
            for i, cr in enumerate(y_vals):
                if math.isfinite(cr) and cr < baseline_cr:
                    first_beat = int(x_months[i])
                    break

            # "Stays below" — first month after which CR is consistently
            # below baseline for the rest of the timeline
            stays_below = None
            if first_beat is not None:
                for i in range(len(y_vals)):
                    if all(
                        math.isfinite(y_vals[j]) and y_vals[j] < baseline_cr
                        for j in range(i, len(y_vals))
                    ):
                        stays_below = int(x_months[i])
                        break

            final_cr = y_vals[-1] if math.isfinite(y_vals[-1]) else float("inf")
            min_cr   = min((v for v in y_vals if math.isfinite(v)), default=float("inf"))

            if math.isfinite(final_cr) and math.isfinite(baseline_cr):
                improvement = (baseline_cr - final_cr) / baseline_cr * 100
                improve_str = f"{improvement:+.2f}%"
            else:
                improve_str = "N/A"

            first_str  = f"Month {first_beat}"  if first_beat  is not None else "Never"
            stays_str  = f"Month {stays_below}" if stays_below is not None else "Never"

            print(f"  {k:>8}-bit │ {first_str:>12} │ {stays_str:>12} │ "
                  f"{final_cr:>9.4f} │ {improve_str:>12} │ {min_cr:>8.4f}")

    print_crossover_report(threat_results, threat_baseline_cr, "Threat-Based")
    print_crossover_report(expo_results,   expo_baseline_cr,   "Randomized EXPO")
    print()

    # ── Charts ────────────────────────────────────────────────────────────────
    print("  Generating charts...")

    plot_robustness_chart(
        x_months, threat_baseline_cr, threat_results, k_bits_list,
        title=f"Oracle Robustness: Threat-Based Trading\n{name} — 5 Year Timeline",
        filename=f"robustness_{prefix}_threat_5yr.png",
        annotate_drops=True,
    )

    plot_robustness_chart(
        x_months, expo_baseline_cr, expo_results, k_bits_list,
        title=f"Oracle Robustness: Randomized EXPO\n{name} — 5 Year Timeline",
        filename=f"robustness_{prefix}_expo_5yr.png",
        annotate_drops=False,
    )

    print(f"  Done.\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    K_BITS = [1, 2, 3, 4, 8]

    DATASETS = [
        {
            "file":      "HistoricalData_1773022846406.csv",
            "price_col": "Close/Last",
            "name":      "Apple Stock",
            "prefix":    "apple",
        },
        {
            "file":      "USD_PHP.csv",
            "price_col": "Close/Last",
            "name":      "USD → PHP",
            "prefix":    "usd_php",
        },
        {
            "file":      "Bitcoin.csv",
            "price_col": "Close/Last",
            "name":      "Bitcoin",
            "prefix":    "bitcoin",
        },
        {
            "file":      "Meta.csv",
            "price_col": "Close/Last",
            "name":      "Meta",
            "prefix":    "meta",
        },
    ]

    for ds in DATASETS:
        df = prepare_df(ds["file"], ds["price_col"])
        if df is not None:
            run_restricted_simulation(
                df, ds["price_col"], K_BITS, ds["name"], ds["prefix"]
            )
        else:
            print(f"  Skipping {ds['name']} — could not load data.")