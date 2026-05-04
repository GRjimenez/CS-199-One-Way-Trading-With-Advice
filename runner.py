"""
runner.py
=========
Tests k-bit geometric-partitioning advice on:
1. Threat-Based (Deterministic, Time-Aware)
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# ── IMPORTS ──
from threat_based import BaseThreatTrader  
from k_bit_advice import KBitThreatTrader


CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# Colours per k-bit level
COLOURS = {
    0: "#6c757d",          # grey   
    1: "#2196F3",          # blue   
    2: "#FF9800",          # orange 
    3: "#4CAF50",          # green
    4: "#9C27B0",   # purple — 4-bit
    5: "#F44336",   # red    — 5-bit
    6: "#00BCD4",   # cyan   — 6-bit
    7: "#795548",   # brown  — 7-bit
    8: "#8BC34A",   # light green — 8-bit  
    "opt": "#E91E63",      # pink   
    "price": "#1a1a2e",    # dark navy 
}

# =============================================================================
# Oracle
# =============================================================================

def oracle_advice_index(true_max, global_m, global_M, k_bits):
    num_intervals = 2 ** k_bits
    ratio = global_M / global_m
    for i in range(num_intervals):
        upper = global_m * (ratio ** ((i + 1) / num_intervals))
        if true_max <= upper or i == num_intervals - 1:
            return i
    return num_intervals - 1

# =============================================================================
# CSV loading
# =============================================================================

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
# Build one trader and collect its trade records
# =============================================================================

def build_and_run(prices, dates, m, M, n, alg_key):
    true_max = float(np.max(prices))
    
    # Parse the algorithm key (e.g., "THREAT_2", "EXPO_3", "RAND_EXPO_1")
    parts = str(alg_key).split("_")
    k_bits = int(parts[-1]) if parts[-1].isdigit() else 0
    idx = oracle_advice_index(true_max, m, M, k_bits) if k_bits > 0 else 0
    
    expected_cash = 0.0

    # ── 1. RANDOMIZED EXPO (100 runs for Expected Value, 1 run for logs) ──
    if "RAND_EXPO" in alg_key:
        all_cashes = []
        sample_trader = None
        for _ in range(100):
            trader = RandomizedExpoTrader(m=m, M=M, n=n, k_bits=k_bits, advice_index=idx)
            for i, p in enumerate(prices):
                trader.trade(float(p), i + 1, str(dates[i]))
            all_cashes.append(trader.cash)
            sample_trader = trader # keep the last one for the logs
        
        expected_cash = np.mean(all_cashes)
        trader = sample_trader 
        m_used, M_used = trader.m, trader.M

    # ── 2. DETERMINISTIC EXPO ──
    elif "EXPO" in alg_key:
        trader = ExpoTrader(m=m, M=M, n=n, k_bits=k_bits, advice_index=idx)
        for i, p in enumerate(prices):
            trader.trade(float(p), i + 1, str(dates[i]))
        expected_cash = trader.cash
        m_used, M_used = trader.m, trader.M

    # ── 3. THREAT-BASED ──
    else:
        if k_bits == 0:
            trader = BaseThreatTrader(n=n, m=m, M=M)
        else:
            trader = KBitThreatTrader(n=n, m=m, M=M, k_bits=k_bits, advice_index=idx)
        for i, p in enumerate(prices):
            trader.trade(float(p), i + 1, str(dates[i]))
        expected_cash = trader.cash
        m_used, M_used = trader.m, trader.M

    # Format the log days
    for i, rec in enumerate(trader.trades):
        rec["Day"] = i + 1

    return trader, expected_cash, m_used, M_used, k_bits


# =============================================================================
# Chart: price series + sell markers per algorithm
# =============================================================================

def plot_trading_chart(prices, dates, traders_info, opt_day, label, filename):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}, facecolor="#f8f9fa")
    ax_price, ax_cash = axes
    days = np.arange(1, len(prices) + 1)

    # ── Top panel: price + sell markers ──
    ax_price.set_facecolor("#ffffff")
    ax_price.plot(days, prices, color=COLOURS["price"], linewidth=1.5, label=f"{label} Price", zorder=2)
    ax_price.axvline(x=opt_day, color=COLOURS["opt"], linewidth=2, linestyle="--", alpha=0.8, label=f"OPT day ({opt_day})", zorder=3)
    ax_price.scatter([opt_day], [prices[opt_day - 1]], color=COLOURS["opt"], s=120, zorder=5, marker="*")

    for alg_key, c_val, trader, exp_cash, k_bits in traders_info:
        trades_df = pd.DataFrame(trader.trades)
        sells = trades_df[(trades_df["Action"] != "HOLD") & (trades_df["Sold"] > 1e-9)]
        if sells.empty: continue
            
        sell_days = sells["Day"].values
        sell_prices = [prices[d - 1] for d in sell_days]
        sell_shares = sells["Sold"].values
        sizes = 40 + (sell_shares / sell_shares.max()) * 120 if sell_shares.max() > 0 else 60
        
        # Format Legend Label
        label_k = f"{alg_key.replace('_', ' ')} (c={c_val:.3f})"
        if "RAND" in alg_key: label_k += " [Sample Run]"
            
        ax_price.scatter(sell_days, sell_prices, s=sizes, color=COLOURS.get(k_bits, "#000"), zorder=4,
                         label=label_k, alpha=0.85, edgecolors="white", linewidths=0.5)

    ax_price.set_title(f"{label}\nPrice chart with sell decisions", fontsize=13, fontweight="bold", pad=12)
    ax_price.set_ylabel("Price (USD)", fontsize=11)
    ax_price.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_price.grid(True, alpha=0.3, linestyle="--")
    ax_price.set_xlim(1, len(prices))

    # ── Bottom panel: cumulative cash ──
    ax_cash.set_facecolor("#ffffff")
    for alg_key, c_val, trader, exp_cash, k_bits in traders_info:
        trades_df = pd.DataFrame(trader.trades)
        label_k = alg_key.replace('_', ' ')
        
        # If randomized, plot the sample run's actual cash, but note it in the legend
        if "RAND" in alg_key: label_k += " [Sample Run]"
        ax_cash.plot(trades_df["Day"], trades_df["Cash"], color=COLOURS.get(k_bits, "#000"), linewidth=1.5, label=label_k)

    opt_cash_series = np.zeros(len(prices))
    opt_cash_series[opt_day - 1:] = prices[opt_day - 1] * 100.0
    ax_cash.plot(days, opt_cash_series, color=COLOURS["opt"], linewidth=1.5, linestyle="--", label="OPT cash", alpha=0.7)

    ax_cash.set_xlabel("Trading Day", fontsize=11)
    ax_cash.set_ylabel("Cumulative Cash ($)", fontsize=11)
    ax_cash.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_cash.grid(True, alpha=0.3, linestyle="--")
    ax_cash.set_xlim(1, len(prices))

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(CHART_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Chart: competitive ratio trend across all slices
# =============================================================================

def plot_cr_trend(slice_labels, cr_by_alg, k_bits_list, filename):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")
    x = np.arange(len(slice_labels))

    for alg_key in k_bits_list:
        crs = cr_by_alg[alg_key]
        
        # Extract k_bits for color
        parts = str(alg_key).split("_")
        k = int(parts[-1]) if parts[-1].isdigit() else 0
        
        # Line style logic
        if "RAND_EXPO" in alg_key: style, marker = ':', '^'
        elif "EXPO" in alg_key: style, marker = '--', 's'
        else: style, marker = '-', 'o'
            
        label_k = alg_key.replace("_", " ")
        ax.plot(x, crs, marker=marker, color=COLOURS.get(k, "#000"),
                linewidth=2.5, linestyle=style, markersize=8, label=label_k)

    ax.set_xticks(x)
    ax.set_xticklabels(slice_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Competitive Ratio (OPT / ALG) — lower is better", fontsize=11)
    ax.set_title("Competitive Ratio vs Trading Horizon\nUSD - PHP Comparison", fontsize=13, fontweight="bold", pad=12)
    
    # Put legend outside to avoid cluttering lines
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=1.0)

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(CHART_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Main simulation function
# =============================================================================

def run_simulation(df, price_col, label, k_bits_list, slug):
    if df is None or len(df) == 0: return {}

    prices = df[price_col].values
    m, M, n = float(np.min(prices)), float(np.max(prices)), len(prices)
    opt_cash = M * 100.0
    opt_day = int(np.argmax(prices)) + 1  
    dates = df["Date"].astype(str).values if "Date" in df.columns else [f"Day {i+1}" for i in range(n)]

    print(f"\n{'='*85}")
    print(f"  {label}")
    print(f"{'='*85}")
    print(f"  n={n}  |  m=${m:.2f}  |  M=${M:.2f}  |  M/m={M/m:.3f}")
    print(f"  OPT: sell all 100 shares on day {opt_day} at ${M:.2f} → ${opt_cash:,.2f}\n")
    
    print(f"  {'Algorithm':>18}  {'Sub-interval':>22}  {'Theor. c':>10}  {'Avg/Final Cash':>15}  {'Comp. ratio':>12}")
    print("  " + "-" * 85)
    
    traders_info = []   
    cr_results = {}

    for alg_key in k_bits_list:
        trader, exp_cash, m_used, M_used, k_bits = build_and_run(prices, dates, m, M, n, alg_key)

        cr = opt_cash / exp_cash if exp_cash > 1e-9 else float("inf")
        c_val = trader.c
        traders_info.append((alg_key, c_val, trader, exp_cash, k_bits))
        cr_results[alg_key] = cr

        # Format label
        interval_str = f"[{m_used:.2f}, {M_used:.2f}]"
        print(f"  {alg_key:>18}  {interval_str:>22}  {c_val:>10.4f}  ${exp_cash:>14,.2f}  {cr:>12.4f}")

    # ── TRADE LOGS ──
    print(f"\n  {'─'*85}")
    print(f"  TRADE LOGS  (★ = OPT day — day {opt_day} at ${M:.2f})")
    print(f"  Note: Randomized EXPO shows 1 sample run's trades, but table uses 100-run average.")

    for alg_key, c_val, trader, exp_cash, k_bits in traders_info:
        trades_df = pd.DataFrame(trader.trades)
        active = trades_df[trades_df["Action"] != "HOLD"]
        m_used, M_used = getattr(trader, 'm', m), getattr(trader, 'M', M)

        print(f"\n  ── {alg_key}  |  c={c_val:.4f}  |  interval=[${m_used:.2f}, ${M_used:.2f}]  |  {len(active)} sell(s) across {n} days")
        print(f"  {'Day':>5}  {'Date':>12}  {'Price':>9}  {'Action':<32}  {'Sold':>8}  {'Cash':>12}  {'Shares Left':>11}")
        print("  " + "-" * 105)

        # Show active trades (or all if very short)
        rows = trades_df if n <= 50 else active

        for _, row in rows.iterrows():
            action = row["Action"]
            day_num = int(row["Day"])
            opt_marker = " ★" if day_num == opt_day else "  "
            sold_str = f"{row['Sold']:>8.4f}" if action != "HOLD" else f"{'—':>8}"
            print(
                f"  {day_num:>4}{opt_marker}  {str(row['Date']):>12}  "
                f"${row['Price']:>8.2f}  {action:<32}  "
                f"{sold_str}  ${row['Cash']:>11,.2f}  {row['Shares']:>11.4f}"
            )

    # ── SPLIT AND CHART SEPARATELY ──
    threat_info = [t for t in traders_info if "THREAT" in t[0]]
    det_expo_info = [t for t in traders_info if "EXPO" in t[0] and "RAND" not in t[0]]
    rand_expo_info = [t for t in traders_info if "RAND_EXPO" in t[0]]

    if threat_info:
        plot_trading_chart(prices, dates, threat_info, opt_day, f"{label} (Threat-Based)", f"trading_{slug}_THREAT.png")
    if det_expo_info:
        plot_trading_chart(prices, dates, det_expo_info, opt_day, f"{label} (Det EXPO)", f"trading_{slug}_DET_EXPO.png")
    if rand_expo_info:
        plot_trading_chart(prices, dates, rand_expo_info, opt_day, f"{label} (Rand EXPO - 1 Sample Run)", f"trading_{slug}_RAND_EXPO.png")

    print(f"\n  [✔] Charts saved for {slug}")
    return cr_results

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(__doc__) 

    # Define EXACTLY which algorithms to run
    K_BITS = [
        "THREAT_0", "THREAT_1", "THREAT_2", "THREAT_3", 
        "THREAT_4", "THREAT_5", "THREAT_6", "THREAT_7", "THREAT_8"
    ]

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
            "price_col": "Close/Last", # Ensure this matches your USD_PHP CSV exactly!
            "name": "USD -> PHP",
            "prefix": "usd_php"
        }
    ]

    # Loop through each dataset and run the full suite
    for ds in DATASETS:
        print(f"\n{'*'*90}")
        print(f"  STARTING DATASET: {ds['name']}")
        print(f"{'*'*90}")

        df = prepare_df(ds['file'], ds['price_col'])
        if df is None: 
            print(f"Skipping {ds['name']} due to loading error.")
            continue

        all_slice_labels = []
        all_cr_by_alg = {k: [] for k in K_BITS}

        # ── Day Slices ──
        for n_days in [10, 30, 60, 90, 180]:
            if len(df) < n_days: continue
            
            # Dynamic slug replaces the hardcoded "apple_"
            slug = f"{ds['prefix']}_{n_days}d"
            
            cr_results = run_simulation(df.iloc[:n_days].reset_index(drop=True), ds['price_col'], f"{ds['name']} — {n_days} Days", K_BITS, slug)
            all_slice_labels.append(f"{n_days}d")
            for k in K_BITS: all_cr_by_alg[k].append(cr_results.get(k, float("nan")))

        # ── Year Slices ──
        if "Date" in df.columns:
            years = sorted(df["Date"].dt.year.dropna().unique())
            for k_yr in range(1, min(6, len(years) + 1)):
                yr_slice = years[:k_yr]
                
                # Dynamic slug replaces the hardcoded "apple_"
                slug = f"{ds['prefix']}_{k_yr}yr"
                
                cr_results = run_simulation(df[df["Date"].dt.year.isin(yr_slice)].reset_index(drop=True), ds['price_col'], f"{ds['name']} — {k_yr} Year(s)", K_BITS, slug)
                all_slice_labels.append(f"{k_yr}yr")
                for k in K_BITS: all_cr_by_alg[k].append(cr_results.get(k, float("nan")))

        # ── Trend Chart (Saved specifically for this dataset) ──
        if all_slice_labels:
            trend_filename = f"cr_trend_all_slices_{ds['prefix']}.png"
            plot_cr_trend(all_slice_labels, all_cr_by_alg, K_BITS, trend_filename)
            print(f"\n[✔] Finished {ds['name']}! Master trend chart saved as {trend_filename}")

    print(f"\nAll datasets processed successfully! Check the ./{CHART_DIR}/ folder.")