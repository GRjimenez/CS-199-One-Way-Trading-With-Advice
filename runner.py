# runner.py
import pandas as pd
import numpy as np
from maximum_forecast import BaseThreatTrader
from machine_learning import MaxPricePredictor

def load_and_preprocess(file_path, price_col='Close/Last'):
    df = pd.read_csv(file_path)
    df = df.iloc[::-1].reset_index(drop=True)
    df[price_col] = df[price_col].str.replace(r'[\$,]', '', regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Low'] = df['Low'].str.replace(r'[\$,]', '', regex=True).astype(float)
    df['High'] = df['High'].str.replace(r'[\$,]', '', regex=True).astype(float)
    return df

def run_simulation(file_path, price_col, name, ml_predictor=None, lambda_val=0.5):
    print(f"\n{'='*50}")
    print(f" RUNNING SIMULATION: {name}")
    print(f"{'='*50}")

    df = load_and_preprocess(file_path, price_col)
    n_days = len(df)
    m_min = df['Low'].min()
    M_max = df['High'].max()
    prices = df[price_col].values
    dates = df['Date'].dt.strftime('%m/%d/%Y').values

    print("--- Inputs ---")
    print(f"  n (Total Days) : {n_days}")
    print(f"  m (Floor)      : {m_min}")
    print(f"  M (Ceiling)    : {M_max}")

    trader = BaseThreatTrader(n=n_days, m=m_min, M=M_max, lambda_val=lambda_val, ml_predictor=ml_predictor)
    print(f"  Calculated 'c' : {trader.c:.4f}")

    # Run trading loop
    window_size = 10
    for i in range(n_days):
        past_window = prices[max(0, i-window_size+1):i+1]
        past_window = np.pad(past_window, (window_size - len(past_window), 0), 'edge')
        trader.trade(
            current_price=prices[i],
            day_index=i+1,
            date_str=dates[i],
            past_window=past_window
        )

    print("\n--- Results ---")
    print(f"  Final Cash     : ${trader.cash:,.2f}")
    print(f"  Optimal Cash   : ${M_max * 100:,.2f} (If sold 100 shares at absolute peak)")

    trades_df = pd.DataFrame(trader.trades)
    active_trades = trades_df[trades_df['Sold'] > 0]
    print(f"\n--- Trade Log ({len(active_trades)} Active Trades) ---")
    print(active_trades.to_string(index=False))
    return trader.cash

if __name__ == "__main__":
    train_file = "AppleData_2016-2023.csv"
    test_files = ["AppleData_2024.csv", "AppleData_2025.csv", "AppleData_2026.csv"]

    # Train ML predictor
    train_df = load_and_preprocess(train_file)
    ml_predictor = MaxPricePredictor(n_future_days=10)
    ml_predictor.fit(train_df)

    import matplotlib.pyplot as plt

    lambda_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results_cash = {}   # Final cash results
    results_ratio = {}  # Competitive ratio results

    for test_file in test_files:
        cash_results = []
        ratio_results = []

        # Preload to compute optimal cash
        df = load_and_preprocess(test_file)
        M_max = df['High'].max()
        optimal_cash = M_max * 100  # max achievable cash

        for lam in lambda_values:
            final_cash = run_simulation(
                file_path=test_file,
                price_col='Close/Last',
                name=f"{test_file} (λ={lam})",
                ml_predictor=ml_predictor,
                lambda_val=lam
            )

            cash_results.append(final_cash)
            ratio_results.append(optimal_cash / final_cash)  # competitive ratio

        results_cash[test_file] = cash_results
        results_ratio[test_file] = ratio_results

    # ---- Plot 1: Final Cash vs Lambda ----
    plt.figure(figsize=(10, 6))
    for file_name, cash_values in results_cash.items():
        plt.plot(lambda_values, cash_values, marker='o', label=file_name)
    plt.title("Final Cash vs Lambda Value")
    plt.xlabel("Lambda (λ)")
    plt.ylabel("Final Cash ($)")
    plt.xticks(lambda_values)
    plt.grid(True)
    plt.legend()
    plt.show()

    # ---- Plot 2: Competitive Ratio vs Lambda ----
    plt.figure(figsize=(10, 6))
    for file_name, ratio_values in results_ratio.items():
        plt.plot(lambda_values, ratio_values, marker='o', label=file_name)
    plt.title("Competitive Ratio (Optimal / Final Cash) vs Lambda")
    plt.xlabel("Lambda (λ)")
    plt.ylabel("Optimal Cash / Final Cash (Competitive Ratio)")
    plt.xticks(lambda_values)
    plt.grid(True)
    plt.legend()
    plt.show()