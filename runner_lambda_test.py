import pandas as pd
from bounded_advice_trader import BoundedAdviceTrader
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

def run_simulation(file_path, price_col, name, lambda_trust):
    print(f"\n{'='*50}")
    print(f" RUNNING SIMULATION: {name} (λ={lambda_trust})")
    print(f"{'='*50}")
    
    # Load CSV
    df = pd.read_csv(file_path)
    for col in ['Close/Last', 'Low', 'High']:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
    df = df.head(10)
    
    n_days = len(df)
    m_min = df['Low'].min()
    M_max = df['High'].max()
    prices = df[price_col].values
    dates = df['Date'].values
    
    print(f"--- Inputs ---")
    print(f"  n (Total Days) : {n_days}")
    print(f"  m (Floor)      : {m_min}")
    print(f"  M (Ceiling)    : {M_max}")
    
    # Initialize trader
    trader = BoundedAdviceTrader(
        n=n_days,
        m=m_min,
        M=M_max,
        initial_shares=100,
        k_bits=2,           # number of advice blocks
        advice_block=0,     # which block contains advice
        lambda_trust=lambda_trust
    )
    
    print(f"  Calculated 'c' : {trader.c:.4f}")
    
    # Run trading loop
    for i in range(n_days):
        trader.trade(current_price=prices[i], day_index=i+1, date_str=dates[i])
    
    # Results
    print("\n--- Results ---")
    print(f"  Final Cash     : ${trader.cash:,.2f}")
    print(f"  Optimal Cash   : ${M_max * 100:,.2f} (If sold 100 shares at absolute peak)")
    
    trades_df = pd.DataFrame(trader.trades)
    active_trades = trades_df[trades_df['Action'] != 'HOLD']
    
    print(f"\n--- Trade Log ({len(active_trades)} Active Trades) ---")
    print(active_trades.to_string(index=False))
    
    return trader.cash  # optionally return cash for later plotting

# === MAIN EXECUTION ===
if __name__ == "__main__":
    
    lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    file_path = get_path('AppleData.csv')
    
    results = {}
    for lam in lambda_values:
        cash = run_simulation(
            file_path=file_path,
            price_col='Close/Last',
            name='Apple Dataset (1 Week Test)',
            lambda_trust=lam
        )
        results[lam] = cash
    
    print("\n=== Summary of Results ===")
    for lam, cash in results.items():
        print(f"λ={lam:.2f} -> Final Cash: ${cash:,.2f}")