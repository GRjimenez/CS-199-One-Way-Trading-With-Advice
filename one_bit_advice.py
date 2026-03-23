import math
from scipy.optimize import brentq

class OneBitTwoSearchTrader:
    def __init__(self, m, M, advice_bit, n=2, lam=1, initial_shares=100.0):
        self.n = int(n)          # ADD
        self.m = float(m)
        self.M = float(M)
        self.advice_bit = advice_bit
        self.lam = lam           # ADD
        
        self.initial_shares = initial_shares
        self.shares = initial_shares
        self.cash = 0.0
        self.base_cash = 0.0     # ADD - track separately
        self.trades = []
        
        # CHANGE these two lines
        self.advice_shares = lam * initial_shares
        self.base_shares = (1 - lam) * initial_shares
        
        self.max_seen = float('-inf')  # ADD

        # ADD competitive ratio
        try:
            self.c = brentq(
                lambda r: r - self.n * (max(0, 1 - (self.m * (r-1)) / (self.M - self.m)))**(1.0/self.n),
                1.000001,
                self.M / self.m
            )
        except ValueError as e:
            raise ValueError(f"Could not compute competitive ratio: {e}")
        
        # KEEP threshold and r_prices exactly as they are
        self.threshold = math.sqrt(self.M * self.m)
        
        if self.advice_bit == 0:
            self.r_prices = [
                math.sqrt(self.m * self.threshold), 
                self.threshold
            ]
        else:
            r2_1 = self.threshold 
            term1 = 4 * math.sqrt((self.M**3) * self.m)
            term2 = 5 * self.M * self.m
            r2_2 = 0.5 * (math.sqrt(term1 + term2) - math.sqrt(self.M * self.m))
            self.r_prices = [r2_1, r2_2]

        self.current_target_index = 0
        self.predicted_price = self.r_prices[-1]
        self.advice_executed = False

    def trade(self, current_price, day_index, n_days, date_str):
        action = "HOLD"
        trade_amt = 0.0

        is_new_max = current_price > self.max_seen
        if is_new_max:
            self.max_seen = current_price

        # Rule 1: Final day dump
        if day_index == n_days:
            if self.shares > 0:
                trade_amt = self.shares
                action = "SELL_ALL (Last Day)"

        elif is_new_max and self.shares > 0:

            # Unlock advice portion when predicted price is hit
            if not self.advice_executed and current_price >= self.predicted_price:
                self.advice_executed = True

            # Before advice unlocked: threat-based runs on base_shares only
            # After advice unlocked: threat-based runs on full portfolio
            if self.advice_executed:
                available_shares = self.shares
                available_cash = self.cash
            else:
                available_shares = self.base_shares
                available_cash = self.base_cash

            # One threat-based formula on available portion
            numerator = (current_price * available_shares) - self.c * (available_cash + available_shares * self.m)
            denominator = self.c * (current_price - self.m)

            if denominator > 0:
                s_i = numerator / denominator
                trade_amt = max(0.0, min(s_i, available_shares))

                if trade_amt > 1e-4:
                    action = "SELL Threat-Based (Advice Unlocked)" if self.advice_executed else "SELL Threat-Based"

        if trade_amt > 0:
            self.shares -= trade_amt
            self.cash += trade_amt * current_price
            # Only update base tracking if advice not yet unlocked
            if not self.advice_executed:
                self.base_shares -= trade_amt
                self.base_cash += trade_amt * current_price

        self.trades.append({
            "Date": date_str,
            "Price": current_price,
            "Action": action,
            "Sold": trade_amt,
            "Cash": self.cash,
            "Shares": self.shares
        })