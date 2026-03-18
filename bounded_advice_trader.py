import numpy as np
from scipy.optimize import brentq


class BoundedAdviceTrader:
    def __init__(self, n, m, M,
                 advice_block,
                 k_bits,
                 lambda_trust=0.5,
                 initial_shares=100.0):

        self.n = int(n)
        self.m = float(m)
        self.M = float(M)
        self.initial_shares = float(initial_shares)

        # ---- Competitive ratio c (same as BaseThreatTrader) ----
        try:
            self.c = brentq(
                lambda r: r - self.n * (max(0, 1 - (self.m * (r - 1)) / (self.M - self.m)))**(1.0/self.n),
                1.000001,
                self.M / self.m
            )
        except Exception:
            self.c = self.M / self.m

        # ---- Advice configuration ----
        self.k_bits = k_bits
        self.num_blocks = 2 ** k_bits
        self.block_size = self.n // self.num_blocks

        self.advice_block = advice_block

        self.block_start = advice_block * self.block_size
        self.block_end = self.block_start + self.block_size

        # ---- Share splitting ----
        self.lambda_trust = lambda_trust

        self.advice_shares = self.initial_shares * lambda_trust
        self.robust_shares = self.initial_shares * (1 - lambda_trust)

        self.total_shares = self.initial_shares
        self.cash = 0.0
        self.max_seen = 0.0

        self.trades = []

    def trade(self, current_price, day_index, date_str):

        action = "HOLD"
        trade_amt = 0.0

        day_zero_index = day_index - 1

        # ===============================
        # 1️⃣ Advice Portion (bounded)
        # ===============================
        if (self.block_start <= day_zero_index < self.block_end) \
                and self.advice_shares > 0:

            # Sell advice shares at first new max inside block
            if current_price > self.max_seen:
                trade_amt += self.advice_shares
                self.advice_shares = 0.0
                action = "SELL (Bounded Advice Block)"

        # ===============================
        # 2️⃣ Robust Threat-Based Portion
        # ===============================
        is_new_max = current_price > self.max_seen
        if is_new_max:
            self.max_seen = current_price

        # Last day rule
        if day_index == self.n:
            trade_amt += self.advice_shares + self.robust_shares
            self.advice_shares = 0.0
            self.robust_shares = 0.0
            action = "SELL_ALL (Last Day)"

        elif is_new_max and self.robust_shares > 0:

            numerator = (current_price * self.initial_shares) \
                        - self.c * (self.cash + self.robust_shares * self.m)

            denominator = self.c * (current_price - self.m)

            if denominator > 0:
                s_i = numerator / denominator
                s_i = max(0.0, min(s_i, self.robust_shares))

                if s_i > 1e-6:
                    trade_amt += s_i
                    self.robust_shares -= s_i
                    action = "SELL (Threat-Based)"

        # ===============================
        # Execute Trade
        # ===============================
        if trade_amt > 0:
            self.total_shares -= trade_amt
            self.cash += trade_amt * current_price

        self.trades.append({
            "Date": date_str,
            "Price": current_price,
            "Action": action,
            "Sold": trade_amt,
            "Cash": self.cash,
            "Remaining Shares": self.total_shares
        })