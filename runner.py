import pandas as pd
from threat_based import BaseThreatTrader

def run_simulation(file_path, price_col, name):
    print(f"\n{'='*50}")
    print(f" RUNNING SIMULATION: {name}")
    print(f"{'='*50}")
    
    # 1. Load the cleaned CSV Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return
    
    # 2. Extract the inputs required by the algorithm
    n_days = len(df)
    m_min = df['Low'].min()      # Global minimum floor
    M_max = df['High'].max()     # Global maximum ceiling
    prices = df[price_col].values # Sequence of daily closing prices
    dates = df['Date'].values
    
    print("--- Inputs ---")
    print(f"  n (Total Days) : {n_days}")
    print(f"  m (Floor)      : {m_min}")
    print(f"  M (Ceiling)    : {M_max}")
    
    # 3. Initialize the Algorithm (imported from threat_based.py)
    trader = BaseThreatTrader(n=n_days, m=m_min, M=M_max)
    print(f"  Calculated 'c' : {trader.c:.4f}")
    
    # 4. Run the Trading Loop
    for i in range(n_days):
        trader.trade(current_price=prices[i], day_index=i+1, date_str=dates[i])
        
    # 5. Output Results
    print("\n--- Results ---")
    print(f"  Final Cash     : ${trader.cash:,.2f}")
    print(f"  Optimal Cash   : ${M_max * 100:,.2f} (If sold 100 shares at absolute peak)")
    
    # 6. Show the days where it actually made a trade
    trades_df = pd.DataFrame(trader.trades)
    active_trades = trades_df[trades_df['Action'] != 'HOLD']
    
    print(f"\n--- Trade Log ({len(active_trades)} Active Trades) ---")
    print(active_trades.to_string(index=False))

# === MAIN EXECUTION ===
if __name__ == "__main__":
    
    # Run Dataset 1: Apple
    run_simulation(
        file_path='cleaned_HistoricalData_1771208929614.csv', 
        price_col='Close/Last', 
        name='Apple Stock (10 Years)'
    )
    
    # Run Dataset 2: EUR/USD
    run_simulation(
        file_path='cleaned_EUR_USD Historical Data (1).csv', 
        price_col='Price', 
        name='EUR/USD (10 Years)'
    )
    # In your runner.py, at the bottom:
    run_simulation('test_1_scared.csv', 'Close/Last', 'Test 1: Scared Hold')
    run_simulation('test_2_perfect_peak.csv', 'Close/Last', 'Test 2: Perfect Peak')
    run_simulation('test_3_staircase.csv', 'Close/Last', 'Test 3: Staircase')
    run_simulation('test_4_dip_rule1.csv', 'Close/Last', 'Test 4: Dip & Rule 1')
    run_simulation('test_5_stable.csv', 'Close/Last', 'Test 5: Stable Market')