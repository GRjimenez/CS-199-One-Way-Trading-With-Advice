import pandas as pd
import math
from threat_based import BaseThreatTrader
from one_bit_advice import OneBitTwoSearchTrader

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

def run_base_simulation_df(df, price_col, name, global_m, global_M):
    print(f"\n{'='*50}")
    print(f" BASE SIMULATION: {name}")
    print(f"{'='*50}")

    if df is None or len(df) == 0:
        print("No data available for this simulation.")
        return

    n_days = len(df)
    prices = df[price_col].values
    dates = df['Date'].astype(str).values if 'Date' in df.columns else ["" for _ in range(n_days)]

    print("--- Inputs ---")
    print(f"  n (Total Days) : {n_days}")
    print(f"  m (Global Floor)   : {global_m}")
    print(f"  M (Global Ceiling) : {global_M}")

    trader = BaseThreatTrader(n=n_days, m=global_m, M=global_M)
    print(f"  Calculated 'c' : {trader.c:.4f}")

    for i in range(n_days):
        trader.trade(current_price=prices[i], day_index=i+1, date_str=dates[i])

    print("\n--- Results ---")
    print(f"  Final Cash     : ${trader.cash:,.2f}")
    
    # Calculate optimal cash based on the actual peak of THIS period
    period_max = df[price_col].max()
    print(f"  Optimal Cash   : ${period_max * 100:,.2f} (If sold 100 shares at period peak)")

    trades_df = pd.DataFrame(trader.trades)
    active_trades = trades_df[trades_df['Action'] != 'HOLD']

    print(f"\n--- Trade Log ({len(active_trades)} Active Trades) ---")
    print(active_trades.to_string(index=False))

def run_one_bit_simulation_df(df, price_col, name, global_m, global_M):
    print(f"\n{'='*50}")
    print(f" 1-BIT ADVICE SIMULATION: {name}")
    print(f"{'='*50}")

    if df is None or len(df) == 0:
        print("No data available.")
        return

    n_days = len(df)
    prices = df[price_col].values
    dates = df['Date'].astype(str).values if 'Date' in df.columns else ["" for _ in range(n_days)]

    # 1. Oracle determines the actual max of THIS specific time period
    actual_slice_max = df[price_col].max()

    # 2. Oracle calculates the threshold based on GLOBAL bounds
    threshold = math.sqrt(global_M * global_m)

    # 3. Oracle provides advice (1 if period max crosses threshold, 0 if it doesn't)
    advice_bit = 1 if actual_slice_max >= threshold else 0

    print("--- Inputs ---")
    print(f"  n (Total Days) : {n_days}")
    print(f"  Calculated Threshold: ${threshold:.2f}")
    print(f"  Actual Period Max   : ${actual_slice_max:.2f}")
    print(f"  Advice Bit Given    : {advice_bit}")

    # Initialize trader with the advice bit
    trader = OneBitTwoSearchTrader(n=n_days, m=global_m, M=global_M, advice_bit=advice_bit)

    print("\n--- Reservation Targets ---")
    print(f"  Target 1: ${trader.r_prices[0]:.2f}")
    print(f"  Target 2: ${trader.r_prices[1]:.2f}")

    # Run Trading Loop
    for i in range(n_days):
        trader.trade(current_price=prices[i], day_index=i+1, n_days=n_days, date_str=dates[i])

    print("\n--- Results ---")
    print(f"  Final Cash     : ${trader.cash:,.2f}")
    print(f"  Optimal Cash   : ${actual_slice_max * 100:,.2f} (If sold 100 shares at period peak)")

    trades_df = pd.DataFrame(trader.trades)
    active_trades = trades_df[trades_df['Action'] != 'HOLD']

    print(f"\n--- Trade Log ({len(active_trades)} Active Trades) ---")
    print(active_trades.to_string(index=False))


if __name__ == "__main__":
    file_path = 'HistoricalData_1773022846406.csv'
    price_col = 'Close/Last'
    dataset_name = 'Apple Stock'

    df = prepare_df(file_path, price_col)
    if df is None:
        raise SystemExit(1)

    # ==========================================
    # 1. SIMULATION: FIRST 10 DAYS
    # ==========================================
    k_days = 10
    df_10 = df.iloc[:k_days].reset_index(drop=True) if len(df) >= k_days else df.copy()
    label_10 = f"{dataset_name} — First {min(k_days, len(df))} Day(s)"
    
    # THE ACADEMIC FIX (Choice B): Find the true, exact bounds of this specific 10-day slice
    local_m_10 = df_10[price_col].min()
    local_M_10 = df_10[price_col].max()
    
    run_base_simulation_df(df_10, price_col, label_10, local_m_10, local_M_10)
    run_one_bit_simulation_df(df_10, price_col, label_10, local_m_10, local_M_10)

    # ==========================================
    # 2. SIMULATIONS: FIRST 1, 2, AND 3 YEARS
    # ==========================================
    if 'Date' in df.columns and df['Date'].notna().any():
        years = sorted(df['Date'].dt.year.dropna().unique())
        
        # We only want to run exactly 1, 2, and 3 years
        max_k = min(3, len(years)) 
        
        for k in range(1, max_k + 1):
            years_slice = years[:k]
            # Slice the dataframe to only include the current year(s)
            df_slice = df[df['Date'].dt.year.isin(years_slice)].reset_index(drop=True)
            label_yr = f"{dataset_name} — First {k} Year(s) ({years_slice[0]}-{years_slice[-1]})"
            
            # THE ACADEMIC FIX (Choice B): Find the true, exact bounds of this specific multi-year slice
            local_m = df_slice[price_col].min()
            local_M = df_slice[price_col].max()
            
            # Feed the true bounds into the algorithm to test its theoretical behavior
            run_base_simulation_df(df_slice, price_col, label_yr, local_m, local_M)
            run_one_bit_simulation_df(df_slice, price_col, label_yr, local_m, local_M)
    else:
        print("Error: Could not parse dates to run yearly simulations.")