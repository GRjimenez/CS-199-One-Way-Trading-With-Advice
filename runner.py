import pandas as pd
from threat_based import BaseThreatTrader

def prepare_df(file_path, price_col):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return None

    # Chronological order (oldest -> newest)
    df = df.iloc[::-1].reset_index(drop=True)

    # Normalize numeric columns: strip $ and commas, convert to numeric
    num_cols = ['Low', 'High', 'Open', 'Close/Last', 'Price']
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r'[\$,]', '', regex=True)
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing essential numeric data
    df = df.dropna(subset=[col for col in ['Low', 'High', price_col] if col in df.columns])

    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    return df

def run_simulation_df(df, price_col, name):
    print(f"\n{'='*50}")
    print(f" RUNNING SIMULATION: {name}")
    print(f"{'='*50}")

    if df is None or len(df) == 0:
        print("No data available for this simulation.")
        return

    n_days = len(df)
    m_min = df[price_col].min()      
    M_max = df[price_col].max()
    prices = df[price_col].values
    dates = df['Date'].astype(str).values if 'Date' in df.columns else ["" for _ in range(n_days)]

    print("--- Inputs ---")
    print(f"  n (Total Days) : {n_days}")
    print(f"  m (Floor)      : {m_min}")
    print(f"  M (Ceiling)    : {M_max}")

    trader = BaseThreatTrader(n=n_days, m=m_min, M=M_max)
    print(f"  Calculated 'c' : {trader.c:.4f}")

    for i in range(n_days):
        trader.trade(current_price=prices[i], day_index=i+1, date_str=dates[i])

    print("\n--- Results ---")
    print(f"  Final Cash     : ${trader.cash:,.2f}")
    print(f"  Optimal Cash   : ${M_max * 100:,.2f} (If sold 100 shares at absolute peak)")

    trades_df = pd.DataFrame(trader.trades)
    active_trades = trades_df[trades_df['Action'] != 'HOLD']

    print(f"\n--- Trade Log ({len(active_trades)} Active Trades) ---")
    print(active_trades.to_string(index=False))

if __name__ == "__main__":
    file_path = 'HistoricalData_1773022846406.csv'
    price_col = 'Close/Last'
    dataset_name = 'Apple Stock (Dataset)'

    df = prepare_df(file_path, price_col)
    # --- First 10 days simulation ---
    k_days = 10
    df_10 = df.iloc[:k_days].reset_index(drop=True) if len(df) >= k_days else df.copy()
    label = f"{dataset_name} — First {min(k_days, len(df))} Day(s)"
    run_simulation_df(df_10, price_col, label)
    if df is None:
        raise SystemExit(1)

    # Determine chronological years present (oldest -> newest)
    if 'Date' in df.columns and df['Date'].notna().any():
        years = sorted(df['Date'].dt.year.dropna().unique())
    else:
        # Fall back: if no parseable dates, treat full dataset as single period
        years = []

    # If there are parseable years, run first 1,2,3 cumulative years
    if years:
        max_k = min(3, len(years))
        for k in range(1, max_k + 1):
            years_slice = years[:k]
            df_slice = df[df['Date'].dt.year.isin(years_slice)].reset_index(drop=True)
            run_simulation_df(
                df_slice,
                price_col,
                f"{dataset_name} — First {k} Year(s) ({years_slice[0]}-{years_slice[-1]})"
            )
    else:
        # No year info: run single simulation on the whole dataset
        run_simulation_df(df, price_col, f"{dataset_name} — Full Dataset")