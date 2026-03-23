import math
from scipy.optimize import brentq

class OneBitTwoSearchTrader:
    def __init__(self, n, m, M, advice_bit, initial_shares=100.0):
        self.n = int(n)
        self.m = float(m)
        self.M = float(M)
        self.advice_bit = advice_bit
        
        self.initial_shares = initial_shares
        self.shares = initial_shares
        self.cash = 0.0
        self.trades = []
        self.max_seen = float('-inf')

        # Compute natural competitive ratio
        
        try:
            self.c = brentq(
                lambda r: r - self.n * (1 - (max(0, (self.m * (r - 1)) / (self.M - self.m)))**(1.0/self.n)),
                1.000001,
                self.M / self.m
            )
        except ValueError as e:
            raise ValueError(f"Could not compute competitive ratio: {e}")

        self.natural_c = self.c
        self.threshold = math.sqrt(self.M * self.m)

        # Advice modifies c only
        if advice_bit == 0:
            # Weak market: lower c, sell more at each new max
            self.c = max(1.001, self.natural_c * (self.m / self.threshold))
        else:
            # Strong market: raise c, sell less early save for later highs
            # bounded so it never exceeds M/m
            self.c = min(self.M / self.m, self.natural_c * (self.threshold / self.m))

        # Reservation price targets for the two-search one-bit strategy.
        # Provide two price targets so caller (runner.py) can report them.
        # Use conservative targets when advice_bit==0 (weak market),
        # and more aggressive targets when advice_bit==1 (strong market).
        if self.advice_bit == 1:
            # Strong market: aim for threshold then the period ceiling
            self.r_prices = [self.threshold, self.M]
        else:
            # Weak market: start at the floor then the threshold
            self.r_prices = [self.m, self.threshold]

    def trade(self, current_price, day_index, n_days, date_str):
        action = "HOLD"
        trade_amt = 0.0

        is_new_max = current_price > self.max_seen
        if is_new_max:
            self.max_seen = current_price

        # Last day dump
        if day_index == n_days:
            if self.shares > 0:
                trade_amt = self.shares
                action = "SELL_ALL (Last Day)"

        elif is_new_max and self.shares > 0:
            # Pure threat-based, c adjusted by advice
            numerator = (current_price * self.initial_shares) - self.c * (self.cash + self.shares * self.m)
            denominator = self.c * (current_price - self.m)

            if denominator > 0:
                s_i = numerator / denominator
                trade_amt = max(0.0, min(s_i, self.shares))

                if trade_amt > 1e-4:
                    action = f"SELL Threat-Based (Bit={self.advice_bit})"

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