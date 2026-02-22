import numpy as np
from scipy.optimize import brentq

class BaseThreatTrader:
    def __init__(self, n, m, M, initial_shares=100.0):
        self.n = int(n)
        self.m = float(m)
        self.M = float(M)
        self.initial_shares = initial_shares # Added this
        self.shares = initial_shares
        self.cash = 0.0
        self.max_seen = 0.0
        self.trades = []
        
        try:
            self.c = brentq(
                lambda r: r - self.n * (max(0, 1 - (self.m * (r - 1)) / (self.M - self.m)))**(1.0/self.n), 
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
            
        if day_index == self.n:
            trade_amt = self.shares
            action = "SELL_ALL (Last Day)"
            
        elif is_new_max and self.shares > 0:
            # THE CORRECTED FORMULA
            # We must multiply the current_price by the initial_shares (X_0)
            numerator = (current_price * self.initial_shares) - self.c * (self.cash + self.shares * self.m)
            denominator = self.c * (current_price - self.m)
            
            if denominator > 0:
                s_i = numerator / denominator
                trade_amt = max(0.0, min(s_i, self.shares))
                
                if trade_amt > 1e-4:
                    action = "SELL (New Max)"

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