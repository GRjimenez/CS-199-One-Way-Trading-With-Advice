import numpy as np
from scipy.optimize import brentq

class KBitThreatTrader:
    def __init__(self, n, m, M, k_bits, advice_index, initial_shares=100.0):
        self.n = int(n)
        self.global_m = float(m)
        self.global_M = float(M)
        self.k_bits = int(k_bits)
        self.advice_index = int(advice_index)
        
        self.initial_shares = initial_shares
        self.shares = initial_shares
        self.cash = 0.0
        self.max_seen = 0.0
        self.trades = []
        
        # ==========================================
        # THE ORACLE TRANSLATION (Geometric Partitioning)
        # ==========================================
        num_intervals = 2 ** self.k_bits
        ratio = self.global_M / self.global_m
        
        # Shrink the global bounds to the specific sub-interval
        self.m = self.global_m * (ratio ** (self.advice_index / num_intervals))
        self.M = self.global_m * (ratio ** ((self.advice_index + 1) / num_intervals))
        
        # ==========================================
        # CALCULATE OPTIMIZED c
        # ==========================================
        try:
            self.c = brentq(
                lambda r: r - self.n * (1 - (max(0, (self.m * (r - 1)) / (self.M - self.m)))**(1.0/self.n)), 
                1.000001, 
                self.M / self.m
            )
        except Exception:
            self.c = self.M / self.m 
                
    def trade(self, current_price, day_index, date_str):
        action = "HOLD"
        trade_amt = 0.0

        is_new_max = current_price > self.max_seen
        if is_new_max:
            self.max_seen = current_price

        # Last day dump
        if day_index == self.n:
            if self.shares > 0:
                trade_amt = self.shares
                action = "SELL_ALL (Last Day)"

        # Standard Threat-Based Fractional Selling
        elif is_new_max and self.shares > 0:
            numerator = (current_price * self.initial_shares) - self.c * (self.cash + self.shares * self.m)
            denominator = self.c * (current_price - self.m)

            if denominator > 0:
                s_i = numerator / denominator
                trade_amt = max(0.0, min(s_i, self.shares))

                if trade_amt > 1e-4:
                    action = f"SELL Threat-Based ({self.k_bits}-Bit Advice)"

        if trade_amt > 0:
            self.shares -= trade_amt
            self.cash += trade_amt * current_price

        self.trades.append({
            "Date": date_str,
            "Price": current_price,
            "Action": action,
            "Sold": trade_amt,
            "Cash": self.cash,
            "Shares": self.shares
        })