"""
runner.py
=========
Tests k-bit geometric-partitioning advice (Clemente et al. 2022) on the
El-Yaniv / Heinsbroek threat-based one-way trading algorithm.

HOW THIS WORKS — summary of the whole pipeline:
================================================

1. PROBLEM (El-Yaniv et al. 2001)
   --------------------------------
   One-way trading: you start with 100 shares and a known price range [m, M].
   Each day a price is revealed. You decide how many shares to sell (irrevocable).
   Goal: maximise total cash. OPT would sell everything on the single best day.

2. THREAT-BASED ALGORITHM (0-bit baseline)
   ------------------------------------------
   Instead of guessing the best day, the algorithm asks:
   "What is the WORST that can happen from here — if all future prices drop to m?"
   It sells just enough shares today so that even in that worst case, the ratio
   OPT / ALG stays ≤ c.  This gives a provably c-competitive algorithm.

   c is the unique fixed point of (El-Yaniv eq. 2):
       c = n * (1 - (m(c-1) / (M-m))^(1/n))
   Larger M/m or larger n → larger c (harder problem, worse guarantee).

3. K-BIT ADVICE (Clemente et al. 2022)
   ---------------------------------------
   The oracle knows the true maximum price p* and encodes WHICH of 2^k
   geometrically-equal sub-intervals of [m, M] contains p*.
   This k-bit message lets the algorithm replace [m, M] with a tighter
   interval [m', M'], compute a smaller c', and trade more aggressively.

   With k=0: no advice, full [m, M].
   With k=1: oracle says "peak is in upper or lower half" → interval halved.
   With k=2: oracle says "which quarter" → interval quartered.
   With k=3: oracle says "which eighth" → interval eighth-ed.

   Each extra bit roughly halves the sub-interval → diminishing returns per bit.

4. OPTIMAL SOLUTION (OPT)
   -------------------------
   The offline optimal simply sells all 100 shares on the day with the
   highest price. We show this as a benchmark — no online algorithm can
   match it without foreknowledge.

5. EXPERIMENTS
   -------------
   We test on real Apple stock data across 8 time slices:
   10, 30, 60, 90, 180 days then 1, 2, 3 years.
   Longer slices → higher M/m → more room for advice to help.

Outputs per slice:
   - Summary table: competitive ratio and % improvement per k
   - Trade log: which days each algorithm actually sold on
   - Chart: price series + sell markers for each algorithm + OPT day marked
   - Competitive ratio trend chart across all slices
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from threat_based import BaseThreatTrader
from k_bit_advice import KBitThreatTrader

# Output folder for charts
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# Colours per k-bit level — consistent across all charts
COLOURS = {
    0: "#6c757d",   # grey   — base
    1: "#2196F3",   # blue   — 1-bit
    2: "#FF9800",   # orange — 2-bit
    3: "#4CAF50",   # green  — 3-bit
    "opt": "#E91E63",  # pink — OPT
    "price": "#1a1a2e", # dark navy — price line
}


# =============================================================================
# Oracle
# =============================================================================

def oracle_advice_index(true_max, global_m, global_M, k_bits):
    """
    Returns the index of the geometric sub-interval containing true_max.
    Sub-interval i: [global_m * ratio^(i/N),  global_m * ratio^((i+1)/N)]
    where N = 2^k_bits.
    """
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

    df = df.iloc[::-1].reset_index(drop=True)  # oldest first

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

def build_and_run(prices, dates, m, M, n, k_bits):
    """
    Instantiate the right trader, run it, return (trader, m_used, M_used).
    """
    true_max = float(np.max(prices))

    if k_bits == 0:
        trader = BaseThreatTrader(n=n, m=m, M=M)
        m_used, M_used = m, M
    else:
        idx = oracle_advice_index(true_max, m, M, k_bits)
        trader = KBitThreatTrader(n=n, m=m, M=M, k_bits=k_bits, advice_index=idx)
        m_used, M_used = trader.m, trader.M

    for i, p in enumerate(prices):
        trader.trade(current_price=float(p), day_index=i + 1, date_str=str(dates[i]))

    for i, rec in enumerate(trader.trades):
        rec["Day"] = i + 1

    return trader, m_used, M_used


# =============================================================================
# Chart: price series + sell markers per algorithm + OPT day
# =============================================================================

def plot_trading_chart(prices, dates, k_bits_list, traders_info, opt_day, label, filename):
    """
    traders_info: list of (k, c_val, trader)
    opt_day     : 1-indexed day of the true price peak
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#f8f9fa"
    )
    ax_price, ax_cash = axes
    days = np.arange(1, len(prices) + 1)

    # ── Top panel: price + sell markers ──────────────────────────────────────
    ax_price.set_facecolor("#ffffff")
    ax_price.plot(days, prices, color=COLOURS["price"], linewidth=1.5,
                  label="Apple price", zorder=2)

    # OPT vertical line — the best day to have sold everything
    ax_price.axvline(x=opt_day, color=COLOURS["opt"], linewidth=2,
                     linestyle="--", alpha=0.8, label=f"OPT day (day {opt_day})", zorder=3)
    ax_price.scatter([opt_day], [prices[opt_day - 1]],
                     color=COLOURS["opt"], s=120, zorder=5, marker="*")

    # Sell markers per algorithm
    for k, c_val, trader in traders_info:
        trades_df = pd.DataFrame(trader.trades)
        sells = trades_df[
            (trades_df["Action"] != "HOLD") &
            (trades_df["Sold"] > 1e-9)
        ]
        if sells.empty:
            continue
        sell_days = sells["Day"].values
        sell_prices = [prices[d - 1] for d in sell_days]
        sell_shares = sells["Sold"].values

        # Marker size proportional to shares sold
        sizes = 40 + (sell_shares / sell_shares.max()) * 120 if sell_shares.max() > 0 else 60
        label_k = f"{k}-bit (c={c_val:.3f})" if k > 0 else f"0-bit base (c={c_val:.3f})"
        ax_price.scatter(sell_days, sell_prices, s=sizes,
                         color=COLOURS[k], zorder=4,
                         label=label_k, alpha=0.85,
                         edgecolors="white", linewidths=0.5)

    ax_price.set_title(f"{label}\nPrice chart with sell decisions per algorithm",
                       fontsize=13, fontweight="bold", pad=12)
    ax_price.set_ylabel("Price (USD)", fontsize=11)
    ax_price.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_price.grid(True, alpha=0.3, linestyle="--")
    ax_price.set_xlim(1, len(prices))

    # ── Bottom panel: cumulative cash per algorithm ───────────────────────────
    ax_cash.set_facecolor("#ffffff")
    for k, c_val, trader in traders_info:
        trades_df = pd.DataFrame(trader.trades)
        label_k = f"{k}-bit" if k > 0 else "0-bit base"
        ax_cash.plot(trades_df["Day"], trades_df["Cash"],
                     color=COLOURS[k], linewidth=1.5, label=label_k)

    # OPT cash reference (step function — 0 until opt_day, then full value)
    opt_cash_series = np.zeros(len(prices))
    opt_cash_series[opt_day - 1:] = prices[opt_day - 1] * 100.0
    ax_cash.plot(days, opt_cash_series, color=COLOURS["opt"],
                 linewidth=1.5, linestyle="--", label="OPT cash", alpha=0.7)

    ax_cash.set_xlabel("Trading Day", fontsize=11)
    ax_cash.set_ylabel("Cumulative Cash ($)", fontsize=11)
    ax_cash.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_cash.grid(True, alpha=0.3, linestyle="--")
    ax_cash.set_xlim(1, len(prices))

    plt.tight_layout(pad=2.0)
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [chart saved → {path}]")


# =============================================================================
# Chart: competitive ratio trend across all slices
# =============================================================================

def plot_cr_trend(slice_labels, cr_by_k, k_bits_list, filename):
    """
    Line chart showing competitive ratio per k-bit level across all slices.
    Lower is better.
    """
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    x = np.arange(len(slice_labels))

    for k in k_bits_list:
        crs = cr_by_k[k]
        label_k = f"{k}-bit advice" if k > 0 else "0-bit base (no advice)"
        ax.plot(x, crs, marker="o", color=COLOURS[k],
                linewidth=2, markersize=7, label=label_k)

    ax.set_xticks(x)
    ax.set_xticklabels(slice_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Competitive Ratio (OPT / ALG) — lower is better", fontsize=11)
    ax.set_title("Competitive Ratio vs Trading Horizon\nk-bit Advice on Apple Stock",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=1.0)

    plt.tight_layout(pad=2.0)
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [trend chart saved → {path}]")


# =============================================================================
# Main simulation function
# =============================================================================

def run_simulation(df, price_col, label, k_bits_list, slug):
    """
    Run all k-bit traders on a price slice.
    Prints summary table, trade logs, saves trading chart.
    Returns dict of competitive ratios keyed by k for the trend chart.
    """
    if df is None or len(df) == 0:
        print("No data.")
        return {}

    prices = df[price_col].values
    m = float(np.min(prices))
    M = float(np.max(prices))
    n = len(prices)
    opt_cash = M * 100.0
    opt_day = int(np.argmax(prices)) + 1  # 1-indexed day of peak price
    dates = (
        df["Date"].astype(str).values
        if "Date" in df.columns
        else [f"Day {i+1}" for i in range(n)]
    )

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  n={n}  |  m=${m:.2f}  |  M=${M:.2f}  |  M/m={M/m:.3f}")
    print(f"  OPT: sell all 100 shares on day {opt_day} at ${M:.2f} → ${opt_cash:,.2f}")
    print()
    print(f"  {'Advice':>12}  {'Sub-interval':>22}  {'Theor. c':>10}  "
          f"{'ALG cash':>12}  {'Comp. ratio':>12}  {'vs 0-bit':>10}")
    print("  " + "-" * 85)

    base_cr = None
    traders_info = []   # (k, c_val, trader) for charting
    cr_results = {}

    for k in k_bits_list:
        try:
            trader, m_used, M_used = build_and_run(prices, dates, m, M, n, k)
        except Exception as e:
            print(f"  {'ERROR':>12}  k={k}: {e}")
            continue

        alg_cash = trader.cash
        cr = opt_cash / alg_cash if alg_cash > 1e-9 else float("inf")
        c_val = trader.c
        interval_str = f"[{m_used:.2f}, {M_used:.2f}]"

        if k == 0:
            base_cr = cr
            vs_base = "—"
        else:
            if base_cr and math.isfinite(base_cr) and math.isfinite(cr) and cr > 0:
                pct = (base_cr - cr) / base_cr * 100
                vs_base = f"{pct:+.2f}%"
            else:
                vs_base = "N/A"

        advice_label = f"{k}-bit" if k > 0 else "0-bit (base)"
        print(
            f"  {advice_label:>12}  {interval_str:>22}  {c_val:>10.4f}  "
            f"${alg_cash:>11,.2f}  {cr:>12.4f}  {vs_base:>10}"
        )
        traders_info.append((k, c_val, trader))
        cr_results[k] = cr

    # ── Trade logs ───────────────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  TRADE LOGS  (★ = OPT day — day {opt_day} at ${M:.2f})")

    for k, c_val, trader in traders_info:
        trades_df = pd.DataFrame(trader.trades)
        advice_label = f"{k}-bit advice" if k > 0 else "0-bit base (no advice)"
        active = trades_df[trades_df["Action"] != "HOLD"]
        _, m_used, M_used = build_and_run.__doc__ and (None, None, None) or (None, None, None)
        # recover interval from trader object
        m_used = trader.m if hasattr(trader, 'm') else m
        M_used = trader.M if hasattr(trader, 'M') else M

        print(f"\n  ── {advice_label}  |  c={c_val:.4f}  |  "
              f"interval=[${m_used:.2f}, ${M_used:.2f}]  |  "
              f"{len(active)} sell(s) across {n} days")
        print(f"  {'Day':>5}  {'Date':>12}  {'Price':>9}  {'Action':<32}  "
              f"{'Sold':>8}  {'Cash':>12}  {'Shares Left':>11}")
        print("  " + "-" * 98)

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

    # ── Trading chart ────────────────────────────────────────────────────────
    chart_filename = f"trading_{slug}.png"
    plot_trading_chart(prices, dates, k_bits_list, traders_info,
                       opt_day, label, chart_filename)

    return cr_results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    FILE_PATH = "HistoricalData_1773022846406.csv"
    PRICE_COL = "Close/Last"
    NAME      = "Apple Stock"
    K_BITS    = [0, 1, 2, 3]

    print(__doc__)  # print the explanation at the top

    df = prepare_df(FILE_PATH, PRICE_COL)
    if df is None:
        raise SystemExit(1)

    # Collect competitive ratios across all slices for the trend chart
    all_slice_labels = []
    all_cr_by_k = {k: [] for k in K_BITS}

    # ── Day-based slices ──────────────────────────────────────────────────────
    for n_days in [10, 30, 60, 90, 180]:
        if len(df) < n_days:
            print(f"Warning: dataset has fewer than {n_days} days, skipping.")
            continue
        df_slice = df.iloc[:n_days].reset_index(drop=True)
        lbl = f"{NAME} — First {n_days} Days"
        slug = f"apple_{n_days}d"
        cr_results = run_simulation(df_slice, PRICE_COL, lbl, K_BITS, slug)
        all_slice_labels.append(f"{n_days}d")
        for k in K_BITS:
            all_cr_by_k[k].append(cr_results.get(k, float("nan")))

    # ── Yearly slices ─────────────────────────────────────────────────────────
    if "Date" in df.columns and df["Date"].notna().any():
        years = sorted(df["Date"].dt.year.dropna().unique())
        for k_yr in range(1, min(4, len(years) + 1)):
            yr_slice = years[:k_yr]
            df_yr = df[df["Date"].dt.year.isin(yr_slice)].reset_index(drop=True)
            lbl = f"{NAME} — {k_yr} Year(s) ({yr_slice[0]}–{yr_slice[-1]})"
            slug = f"apple_{k_yr}yr"
            cr_results = run_simulation(df_yr, PRICE_COL, lbl, K_BITS, slug)
            all_slice_labels.append(f"{k_yr}yr")
            for k in K_BITS:
                all_cr_by_k[k].append(cr_results.get(k, float("nan")))
    else:
        print("Warning: could not parse dates for yearly slices.")

    # ── Trend chart across all slices ─────────────────────────────────────────
    if all_slice_labels:
        plot_cr_trend(all_slice_labels, all_cr_by_k, K_BITS,
                      "cr_trend_all_slices.png")
        print(f"\nAll charts saved to ./{CHART_DIR}/")