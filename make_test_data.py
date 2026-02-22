import pandas as pd
import numpy as np

def make_csv(filename, prices, m, M):
    # Create 10 days of data
    df = pd.DataFrame({
        'Date': [f'2026-01-{str(i+1).zfill(2)}' for i in range(10)],
        'Close/Last': prices,
        'Low': [m] * 10,  # We fix the global minimum
        'High': [M] * 10  # We fix the global maximum
    })
    df.to_csv(filename, index=False)
    print(f"Created {filename}")

# 1. The Scared Hold (Max is 100, but price never passes 15)
make_csv('test_1_scared.csv', 
         prices=[10, 11, 12, 13, 14, 15, 14, 13, 12, 11], m=10, M=100)

# 2. The Perfect Peak (Hits exactly M on Day 4)
make_csv('test_2_perfect_peak.csv', 
         prices=[10, 20, 50, 100, 20, 15, 12, 11, 10, 10], m=10, M=100)

# 3. The Staircase (Consistent steady growth)
make_csv('test_3_staircase.csv', 
         prices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], m=10, M=100)

# 4. The Dip and Recover (Hits 60, drops, goes back to 50)
make_csv('test_4_dip_rule1.csv', 
         prices=[10, 30, 60, 20, 25, 30, 40, 50, 45, 40], m=10, M=100)

# 5. Stable Market (Tight gap, m=1.0, M=1.2)
make_csv('test_5_stable.csv', 
         prices=[1.0, 1.05, 1.02, 1.10, 1.15, 1.12, 1.18, 1.19, 1.10, 1.05], m=1.0, M=1.2)