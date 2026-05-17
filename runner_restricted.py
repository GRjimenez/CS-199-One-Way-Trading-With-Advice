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

Two separate charts generated (CR and Cash):
    1. Threat-Based Oracle Robustness 
    2. Randomized EXPO Oracle Robustness
    3. Summary Data Table (PNG)

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
    if "Date" in df.columns and df["Date"].notna().any():
        start = df["Date"].dropna().iloc[0]
        end   = start + pd.DateOffset(years=5)
        df_5yr = df[df["Date"] < end].reset_index(drop=True)
        if len(df_5yr) > 50:
            return df_5yr
    return df.head(min(1302, len(df))).reset_index(drop=True)


def month_window_size(month, total_days, total_months=60):
    days_per_month = total_days / total_months
    return min(int(math.ceil(month * days_per_month)), total_days)


# =============================================================================
# Plotting - Line Charts
# =============================================================================

def plot_robustness_chart(x_months, baseline_cr, results_dict,
                          k_bits_list, title, filename,
                          annotate_drops=False):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    ax.axhline(1.0, color="#1a1a2e", linewidth=2.0,
               linestyle=":", alpha=0.7, label="Optimal (CR = 1.0)", zorder=2)
    ax.axhspan(1.0, baseline_cr, facecolor="#4CAF50",
               alpha=0.07, label="_nolegend_", zorder=1)
    ax.axhline(baseline_cr, color=COLOURS["baseline"], linewidth=2.5,
               linestyle="--", zorder=3,
               label=f"Baseline 0-Bit (CR: {baseline_cr:.3f})")

    all_vals = [baseline_cr, 1.0]

    for k in k_bits_list:
        y_vals = results_dict[k]
        colour = COLOURS.get(k, "#000000")
        ax.plot(x_months, y_vals, color=colour, linewidth=2.2,
                label=f"{k}-Bit Advice", alpha=0.92, zorder=4)
        
        finite_vals = [v for v in y_vals if math.isfinite(v)]
        all_vals.extend(finite_vals)

        if annotate_drops:
            for i in range(1, len(y_vals)):
                if math.isfinite(y_vals[i-1]) and math.isfinite(y_vals[i]):
                    if y_vals[i-1] - y_vals[i] > 0.05:
                        ax.scatter(x_months[i], y_vals[i], color=colour,
                                   s=70, marker="v", zorder=5)

    ax.set_xlabel("Oracle Data Fed (Months)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Final Expected Competitive Ratio (Full 5 Years)",
                  fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(np.arange(0, 61, 6))
    ax.set_xlim(1, 60)

    finite_all = [v for v in all_vals if math.isfinite(v)]
    y_top = max(finite_all) * 1.06 if finite_all else baseline_cr * 1.1
    y_bot = max(0.97, min(finite_all) * 0.97) if finite_all else 0.97
    ax.set_ylim(bottom=y_bot, top=y_top)

    plt.tight_layout(pad=2.0)
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()

def plot_cash_chart(x_months, baseline_cash, results_dict, opt_cash,
                          k_bits_list, title, filename):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    ax.axhline(opt_cash, color="#1a1a2e", linewidth=2.0,
               linestyle=":", alpha=0.7, label=f"Optimal Cash (${opt_cash:,.2f})", zorder=2)
    
    # Shade ABOVE the baseline (higher cash is better)
    ax.axhspan(baseline_cash, opt_cash, facecolor="#4CAF50",
               alpha=0.07, label="_nolegend_", zorder=1)
               
    ax.axhline(baseline_cash, color=COLOURS["baseline"], linewidth=2.5,
               linestyle="--", zorder=3,
               label=f"Baseline 0-Bit Cash: ${baseline_cash:,.2f}")

    all_vals = [baseline_cash, opt_cash]

    for k in k_bits_list:
        y_vals = results_dict[k]
        colour = COLOURS.get(k, "#000000")
        ax.plot(x_months, y_vals, color=colour, linewidth=2.2,
                label=f"{k}-Bit Advice", alpha=0.92, zorder=4)
        
        all_vals.extend(y_vals)

    ax.set_xlabel("Oracle Data Fed (Months)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Final Expected Cash ($)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(np.arange(0, 61, 6))
    ax.set_xlim(1, 60)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))

    y_top = opt_cash * 1.05
    y_bot = min(all_vals) * 0.95 if min(all_vals) > 0 else 0
    ax.set_ylim(bottom=y_bot, top=y_top)

    plt.tight_layout(pad=2.0)
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()

# =============================================================================
# Plotting - Summary Table Image
# =============================================================================
def generate_cash_table_image(threat_cash, expo_cash, opt_cash, baseline_t_cash, baseline_e_cash, k_bits_list, name, prefix):
    # Milestones to check (Indices are month - 1)
    milestones = [1, 12, 36, 60] 
    
    fig, ax = plt.subplots(figsize=(12, sum(len(k_bits_list) for _ in range(2)) * 0.5 + 3))
    ax.axis('tight')
    ax.axis('off')

    col_labels = ["Algorithm", "Advice Level"] + [f"Month {m} Cash" for m in milestones]
    table_data = []
    
    # Format string helper
    def f_cash(val): return f"${val:,.2f}"

    # Threat rows
    table_data.append(["Threat-Based", "0-Bit (Baseline)"] + [f_cash(baseline_t_cash)] * len(milestones))
    for k in k_bits_list:
        row = ["Threat-Based", f"{k}-Bit"]
        for m in milestones:
            row.append(f_cash(threat_cash[k][m-1]))
        table_data.append(row)

    # Blank divider row
    table_data.append(["---", "---"] + ["---"] * len(milestones))
    
    # Expo rows
    table_data.append(["Randomized EXPO", "0-Bit (Baseline)"] + [f_cash(baseline_e_cash)] * len(milestones))
    for k in k_bits_list:
        row = ["Randomized EXPO", f"{k}-Bit"]
        for m in milestones:
            row.append(f_cash(expo_cash[k][m-1]))
        table_data.append(row)
        
    # Optimal row at the bottom
    table_data.append(["---", "---"] + ["---"] * len(milestones))
    table_data.append(["OPTIMAL (Hindsight)", "Max Possible"] + [f_cash(opt_cash)] * len(milestones))

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Make headers bold and add colors
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2d3436')
        elif table_data[i-1][0] == "---":
            cell.set_facecolor('#dfe6e9')
            cell.set_text_props(color='#dfe6e9') # Hide the text
        elif i == len(table_data): # Optimal row
            cell.set_facecolor('#ffeaa7')
            cell.set_text_props(weight='bold')

    plt.title(f"Final Cash Milestones: {name}", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, f"{prefix}_cash_summary_table.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    → Saved Table: {path}")


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
    print(f"  Days={n}  |  m=${global_m:.2f}  |  M=${global_M:.2f}  |  M/m={global_M/global_m:.3f} | OPT Cash=${opt_cash:,.2f}")

    # ── Baselines (0-bit, no advice) ──────────────────────────────────────────
    try:
        tb = BaseThreatTrader(n=n, m=global_m, M=global_M)
        for i, p in enumerate(prices):
            tb.trade(float(p), i + 1, "")
        threat_baseline_cash = tb.cash
        threat_baseline_cr = opt_cash / tb.cash if tb.cash > 1e-9 else float("inf")
    except Exception as e:
        print(f"  [!] Threat baseline failed: {e}")
        threat_baseline_cash = 0.0
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
            
    expo_baseline_cash = np.mean(expo_cashes) if expo_cashes else 0.0
    expo_baseline_cr = opt_cash / expo_baseline_cash if expo_baseline_cash > 0 else float("inf")

    print(f"  Threat 0-bit baseline Cash : ${threat_baseline_cash:,.2f} (CR: {threat_baseline_cr:.4f})")
    print(f"  EXPO   0-bit baseline Cash : ${expo_baseline_cash:,.2f} (CR: {expo_baseline_cr:.4f})")
    print(f"  Running {total_months} monthly windows × {len(k_bits_list)} k-bit levels...")

    # ── Monthly sweep ─────────────────────────────────────────────────────────
    x_months      = np.arange(1, total_months + 1)
    
    threat_cr_results = {k: [] for k in k_bits_list}
    expo_cr_results   = {k: [] for k in k_bits_list}
    
    threat_cash_results = {k: [] for k in k_bits_list}
    expo_cash_results   = {k: [] for k in k_bits_list}

    for month in x_months:
        window = month_window_size(month, n, total_months)
        restricted_max = float(np.max(prices[:window]))

        for k in k_bits_list:
            idx = oracle_advice_index(restricted_max, global_m, global_M, k)

            # -- Threat-based --
            try:
                tt = KBitThreatTrader(n=n, m=global_m, M=global_M, k_bits=k, advice_index=idx)
                for i, p in enumerate(prices):
                    tt.trade(float(p), i + 1, "")
                t_cash = tt.cash
                t_cr = opt_cash / t_cash if t_cash > 1e-9 else float("inf")
            except Exception:
                t_cash = threat_baseline_cash
                t_cr = threat_baseline_cr
                
            threat_cash_results[k].append(t_cash)
            threat_cr_results[k].append(t_cr)

            # -- Randomized EXPO --
            rng_k = np.random.default_rng(EXPO_SEED + month * 100 + k)
            e_cashes = []
            for _ in range(EXPO_RUNS):
                try:
                    et = RandomizedExpoTrader(m=global_m, M=global_M, n=n, k_bits=k, advice_index=idx, rng=rng_k)
                    for i, p in enumerate(prices):
                        et.trade(float(p), i + 1, "")
                    e_cashes.append(et.cash)
                except Exception:
                    pass
            e_cash = np.mean(e_cashes) if e_cashes else 0.0
            e_cr = opt_cash / e_cash if e_cash > 0 else float("inf")
            
            expo_cash_results[k].append(e_cash)
            expo_cr_results[k].append(e_cr)

        if month % 10 == 0:
            print(f"    Month {month:2d}/60 done")

    print()
    print("  Generating charts and tables...")

    # Plot CR Charts
    plot_robustness_chart(x_months, threat_baseline_cr, threat_cr_results, k_bits_list,
        title=f"Oracle Robustness (CR): Threat-Based Trading\n{name} — 5 Year Timeline",
        filename=f"robustness_{prefix}_threat_cr_5yr.png", annotate_drops=True)

    plot_robustness_chart(x_months, expo_baseline_cr, expo_cr_results, k_bits_list,
        title=f"Oracle Robustness (CR): Randomized EXPO\n{name} — 5 Year Timeline",
        filename=f"robustness_{prefix}_expo_cr_5yr.png", annotate_drops=False)

    # Plot Cash Charts
    plot_cash_chart(x_months, threat_baseline_cash, threat_cash_results, opt_cash, k_bits_list,
        title=f"Oracle Robustness (Final Cash): Threat-Based Trading\n{name} — 5 Year Timeline",
        filename=f"robustness_{prefix}_threat_cash_5yr.png")
        
    plot_cash_chart(x_months, expo_baseline_cash, expo_cash_results, opt_cash, k_bits_list,
        title=f"Oracle Robustness (Final Cash): Randomized EXPO\n{name} — 5 Year Timeline",
        filename=f"robustness_{prefix}_expo_cash_5yr.png")
        
    # Plot Data Table
    generate_cash_table_image(threat_cash_results, expo_cash_results, opt_cash, threat_baseline_cash, expo_baseline_cash, k_bits_list, name, prefix)

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
    ]

    for ds in DATASETS:
        df = prepare_df(ds["file"], ds["price_col"])
        if df is not None:
            run_restricted_simulation(
                df, ds["price_col"], K_BITS, ds["name"], ds["prefix"]
            )
        else:
            print(f"  Skipping {ds['name']} — could not load data.")