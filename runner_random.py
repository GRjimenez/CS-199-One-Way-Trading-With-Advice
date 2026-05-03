"""
runner_random.py
================
A dedicated Monte Carlo analyzer for the Randomized EXPO algorithm.
Runs the algorithm 1,000 times per data slice to expose the true variance,
expected values, and worst-case scenarios, outputting detailed statistical 
reports and histogram distributions to a dedicated folder.
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# ── IMPORTS ──
from randomized_expo import RandomizedExpoTrader

REPORT_DIR = "reports_random"
os.makedirs(REPORT_DIR, exist_ok=True)

# Colours map strictly to the k-bits
K_COLOURS = {
    0: "#6c757d",  # grey   (0-bit)
    1: "#2196F3",  # blue   (1-bit)
    2: "#FF9800",  # orange (2-bit)
    3: "#4CAF50"   # green  (3-bit)
}

# =============================================================================
# Helper Functions
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

def run_monte_carlo(prices, m, M, n, k_bits, runs=1000):
    true_max = float(np.max(prices))
    idx = oracle_advice_index(true_max, m, M, k_bits) if k_bits > 0 else 0
    
    all_cashes = []
    theoretical_c = 0.0

    for _ in range(runs):
        trader = RandomizedExpoTrader(m=m, M=M, n=n, k_bits=k_bits, advice_index=idx)
        theoretical_c = trader.c  # captures the c value used
        
        # Fast execution (no dates needed for raw simulation)
        for i, p in enumerate(prices):
            trader.trade(float(p), i + 1, "")
            
        all_cashes.append(trader.cash)

    return np.array(all_cashes), theoretical_c

# =============================================================================
# Charting: The Distribution Histogram
# =============================================================================

def plot_distribution_histogram(cr_distributions, opt_cash, label, filename):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    # Plot a histogram for each k-bit level
    for k in [0, 1, 2, 3]:
        crs = cr_distributions[k]
        
        # Calculate weights to show probability/frequency instead of raw counts
        weights = np.ones_like(crs) / len(crs)
        
        ax.hist(crs, bins=40, weights=weights, color=K_COLOURS[k], alpha=0.5, 
                edgecolor=K_COLOURS[k], linewidth=1.2, label=f"EXPO {k}-bit")
        
        # Add a vertical line for the Mean Expected Value
        ax.axvline(np.mean(crs), color=K_COLOURS[k], linestyle='dashed', linewidth=2)

    ax.set_xlabel("Competitive Ratio (OPT / ALG) — Lower is better", fontsize=11)
    ax.set_ylabel("Probability (Frequency)", fontsize=11)
    ax.set_title(f"Randomized EXPO Distribution Profile (1,000 Runs)\n{label}", fontsize=14, fontweight="bold", pad=15)
    
    # Custom legend
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9, title="Algorithm & Mean (Dashed)")
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Detailed Reporting Pipeline
# =============================================================================

def run_detailed_simulation(df, price_col, label, slug, runs=1000):
    if df is None or len(df) == 0: return

    prices = df[price_col].values
    m, M, n = float(np.min(prices)), float(np.max(prices)), len(prices)
    opt_cash = M * 100.0

    print(f"\n{'='*95}")
    print(f"  {label}  |  MONTE CARLO REPORT ({runs:,} Runs)")
    print(f"{'='*95}")
    print(f"  n={n}  |  m=${m:.2f}  |  M=${M:.2f}  |  M/m={M/m:.3f}  |  OPT Cash: ${opt_cash:,.2f}")
    print(f"  {'='*95}")
    
    print(f"  {'Advice':>10} | {'Theor. c':>8} | {'Mean CR':>9} | {'Median CR':>9} | "
          f"{'Best CR':>8} | {'Worst CR':>8} | {'Std Dev':>8} |")
    print("  " + "-" * 88)

    cr_distributions = {}

    for k in [0, 1, 2, 3]:
        # Run the Monte Carlo simulation
        cashes, theor_c = run_monte_carlo(prices, m, M, n, k, runs=runs)
        
        # Calculate Competitive Ratios for all runs
        # Handle divide-by-zero just in case
        crs = np.divide(opt_cash, cashes, out=np.full_like(cashes, np.inf), where=cashes!=0)
        cr_distributions[k] = crs
        
        # Extract Statistics
        mean_cr = np.mean(crs)
        median_cr = np.median(crs)
        best_cr = np.min(crs)   # Lowest CR is best
        worst_cr = np.max(crs)  # Highest CR is worst
        std_dev = np.std(crs)
        
        # Print Detailed Row
        lbl = f"{k}-bit"
        print(f"  {lbl:>10} | {theor_c:>8.3f} | {mean_cr:>9.3f} | {median_cr:>9.3f} | "
              f"{best_cr:>8.3f} | {worst_cr:>8.3f} | {std_dev:>8.3f} |")

    # Generate the Histogram Distribution Chart
    chart_filename = f"dist_{slug}.png"
    plot_distribution_histogram(cr_distributions, opt_cash, label, chart_filename)
    print(f"\n  [✔] Distribution chart saved to ./{REPORT_DIR}/{chart_filename}")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    FILE_PATH = "HistoricalData_1773022846406.csv"
    PRICE_COL = "Close/Last"
    NAME      = "Apple Stock"
    RUNS      = 1000  # Number of Monte Carlo iterations per k-bit

    print(__doc__) 
    df = prepare_df(FILE_PATH, PRICE_COL)
    if df is None: raise SystemExit(1)

    # Day Slices
    for n_days in [10, 30, 60, 90, 180]:
        if len(df) < n_days: continue
        slug = f"apple_{n_days}d"
        run_detailed_simulation(df.iloc[:n_days].reset_index(drop=True), PRICE_COL, f"{NAME} — {n_days} Days", slug, runs=RUNS)

    # Year Slices
    if "Date" in df.columns:
        years = sorted(df["Date"].dt.year.dropna().unique())
        for k_yr in range(1, min(4, len(years) + 1)):
            yr_slice = years[:k_yr]
            slug = f"apple_{k_yr}yr"
            run_detailed_simulation(df[df["Date"].dt.year.isin(yr_slice)].reset_index(drop=True), PRICE_COL, f"{NAME} — {k_yr} Year(s)", slug, runs=RUNS)

    print(f"\nAll Monte Carlo analysis complete! Check the ./{REPORT_DIR}/ folder.")