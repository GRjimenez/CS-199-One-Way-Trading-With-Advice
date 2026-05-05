"""
runner_restricted.py
====================
Tests "Oracle Robustness" by restricting the oracle's foresight.
The oracle is only allowed to see the first X months of the dataset
to generate its advice. The algorithms then trade over the full 
5-year timeline using that potentially incomplete advice.

Outputs two separate line plots:
1. Threat-Based Oracle Robustness
2. Randomized EXPO Oracle Robustness
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── IMPORTS ──
from threat_based import BaseThreatTrader
from k_bit_advice import KBitThreatTrader
from randomized_expo import RandomizedExpoTrader

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# Colours
COLOURS = {
    "baseline": "#E91E63", # Pink for Baseline (0-bit)
    1: "#2196F3",          # Blue
    2: "#FF9800",          # Orange
    3: "#4CAF50",          # Green
    4: "#9C27B0",          # Purple
    8: "#8BC34A",          # Light Green
}

# =============================================================================
# Oracle (Restricted)
# =============================================================================

def oracle_advice_index(restricted_max, global_m, global_M, k_bits):
    """
    Generates advice based ONLY on the maximum price seen in the restricted window.
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
        print(f"Error: file not found — {file_path}")
        return None

    # Reverse to chronological order (oldest to newest)
    df = df.iloc[::-1].reset_index(drop=True)
    
    for c in ["Low", "High", "Open", "Close/Last", "Price"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r"[\$,]", "", regex=True)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[price_col])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df

# =============================================================================
# Plotting Helper
# =============================================================================

def plot_robustness_chart(x_months, baseline_cr, results_dict, k_bits_list, title_label, filename):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#f8f9fa") # Slightly larger for better labels
    ax.set_facecolor("#ffffff")

    # 1. OPT Line (CR=1.0)
    ax.axhline(1.0, color="#1a1a2e", linewidth=2.5, linestyle=":", alpha=0.8, label="Optimal (CR=1.0)")

    # 2. Shaded Region for "Advice Beats Baseline"
    # Shade the area between the perfect 1.0 and the baseline CR
    ax.axhspan(1.0, baseline_cr, facecolor="#4CAF50", alpha=0.08, label="Advice Beats Baseline")

    # Plot Baseline
    ax.axhline(baseline_cr, color=COLOURS["baseline"], linewidth=3, linestyle="--", 
               label=f"Baseline 0-Bit (CR: {baseline_cr:.3f})")

    max_plotted_val = baseline_cr

    # Plot each k-bit level
    for k in k_bits_list:
        y_vals = results_dict[k]
        line_color = COLOURS.get(k, "#000")
        
        ax.plot(x_months, y_vals, color=line_color, linewidth=2.5, 
                label=f"{k}-Bit Advice", alpha=0.9)
        
        # 3. Dynamic Scaling Fix (Find the highest actual value to prevent clipping)
        valid_vals = [v for v in y_vals if v != float('inf') and not math.isnan(v)]
        if valid_vals:
            max_plotted_val = max(max_plotted_val, max(valid_vals))

        # 4. Annotate "Drop Points" for the Threat-Based step-functions
        # (We only do this for Threat-Based to avoid cluttering the noisy EXPO charts)
        if "Threat" in title_label:
            for i in range(1, len(y_vals)):
                # If the CR drops significantly (Oracle saw a new peak)
                if y_vals[i-1] - y_vals[i] > 0.02:
                    ax.scatter(x_months[i], y_vals[i], color=line_color, 
                               s=80, marker='v', zorder=5)

    # 5. Bigger Fonts for Axes Labels and Titles
    ax.set_xlabel("Oracle Data Fed (Months)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Final Expected Competitive Ratio (Full 5 Years)", fontsize=14, fontweight="bold")
    ax.set_title(title_label, fontsize=16, fontweight="bold", pad=15)
    
    # Increase tick label size for readability
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Clean up legend
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="#ccc")
    ax.grid(True, alpha=0.3, linestyle="--")
    
    ax.set_xticks(np.arange(0, 61, 6))
    ax.set_xlim(1, 60)

    # Apply the dynamic Y-axis fix, capping the top with 5% padding so nothing is cut off
    upper_limit = max_plotted_val * 1.05
    ax.set_ylim(bottom=0.98, top=upper_limit)

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(CHART_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Main Simulation
# =============================================================================

def run_restricted_simulation(df, price_col, k_bits_list, name, prefix):
    prices = df[price_col].values
    total_days = len(prices)
    global_m = float(np.min(prices))
    global_M = float(np.max(prices))
    opt_cash = global_M * 100.0

    print(f"\n{'='*80}")
    print(f"  ORACLE ROBUSTNESS TEST: {name}")
    print(f"{'='*80}")
    print(f"  Total Days: {total_days}  |  Global m: ${global_m:.2f}  |  Global M: ${global_M:.2f}")

    # ── 1. Calculate Baselines (0-Bit) ──
    # Threat Baseline
    threat_baseline = BaseThreatTrader(n=total_days, m=global_m, M=global_M)
    for i, p in enumerate(prices):
        threat_baseline.trade(float(p), i + 1, "")
    threat_baseline_cr = opt_cash / threat_baseline.cash if threat_baseline.cash > 0 else float('inf')

    # EXPO Baseline (1000 Runs for stability)
    expo_baseline_cashes = []
    for _ in range(1000):
        expo = RandomizedExpoTrader(m=global_m, M=global_M, n=total_days, k_bits=0)
        for i, p in enumerate(prices):
            expo.trade(float(p), i + 1, "")
        expo_baseline_cashes.append(expo.cash)
    expo_baseline_cr = opt_cash / np.mean(expo_baseline_cashes)

    print(f"  [+] Threat 0-Bit Baseline CR: {threat_baseline_cr:.4f}")
    print(f"  [+] EXPO 0-Bit Baseline CR:   {expo_baseline_cr:.4f}")

    # ── 2. Setup the Monthly Sliding Window ──
    total_months = 60
    days_per_month = total_days / total_months
    x_months = np.arange(1, total_months + 1)
    
    threat_results = {k: [] for k in k_bits_list}
    expo_results = {k: [] for k in k_bits_list}

    # ── 3. Run the Restricted Scenarios ──
    print("  [+] Running restricted oracle scenarios (This will take a moment)...")
    
    for month in x_months:
        window_size = int(math.ceil(month * days_per_month))
        window_size = min(window_size, total_days)
        
        # Oracle finds max price ONLY in this window
        restricted_prices = prices[:window_size]
        restricted_max = float(np.max(restricted_prices))
        
        for k in k_bits_list:
            idx = oracle_advice_index(restricted_max, global_m, global_M, k)
            
            # --- Threat-Based Test ---
            t_trader = KBitThreatTrader(n=total_days, m=global_m, M=global_M, k_bits=k, advice_index=idx)
            for i, p in enumerate(prices):
                t_trader.trade(float(p), i + 1, "")
            t_cr = opt_cash / t_trader.cash if t_trader.cash > 0 else float('inf')
            threat_results[k].append(t_cr)

            # --- EXPO Test (100 Runs per month-slice for speed/stability balance) ---
            e_cashes = []
            for _ in range(1000):
                e_trader = RandomizedExpoTrader(m=global_m, M=global_M, n=total_days, k_bits=k, advice_index=idx)
                for i, p in enumerate(prices):
                    e_trader.trade(float(p), i + 1, "")
                e_cashes.append(e_trader.cash)
            e_cr = opt_cash / np.mean(e_cashes) if np.mean(e_cashes) > 0 else float('inf')
            expo_results[k].append(e_cr)

    # ── 4. Plot the Results ──
    print(f"  [+] Generating Threat robustness chart...")
    plot_robustness_chart(x_months, threat_baseline_cr, threat_results, k_bits_list,
                          f"Oracle Robustness: Threat-Based Trading\n{name} — 5 Year Timeline",
                          f"robustness_{prefix}_threat_5yr.png")

    print(f"  [+] Generating EXPO robustness chart...")
    plot_robustness_chart(x_months, expo_baseline_cr, expo_results, k_bits_list,
                          f"Oracle Robustness: Randomized EXPO\n{name} — 5 Year Timeline",
                          f"robustness_{prefix}_expo_5yr.png")

    print(f"  [✔] Charts successfully saved to ./{CHART_DIR}/\n")


if __name__ == "__main__":
    K_BITS = [1, 2, 3, 4, 8]
    
    # ── Define all your datasets here ──
    DATASETS = [
        {
            "file": "HistoricalData_1773022846406.csv",
            "price_col": "Close/Last",
            "name": "Apple Stock",
            "prefix": "apple"
        },
        {
            "file": "USD_PHP.csv",
            "price_col": "Close/Last", 
            "name": "USD -> PHP",
            "prefix": "usd_php"
        }
    ]

    for ds in DATASETS:
        df = prepare_df(ds['file'], ds['price_col'])
        if df is not None:
            # Make sure we only use the first 5 years of data (~1302 days) to match your previous tests
            df_5yr = df.head(1302).reset_index(drop=True) 
            run_restricted_simulation(df_5yr, ds['price_col'], K_BITS, ds['name'], ds['prefix'])