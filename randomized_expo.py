import math
import random

class RandomizedExpoTrader:
    def __init__(self, m, M, initial_shares=100.0, n=None, k_bits=0, advice_index=0):
        self.global_m = float(m)
        self.global_M = float(M)
        self.initial_shares = float(initial_shares)
        self.shares = float(initial_shares)
        self.cash = 0.0
        self.trades = []
        self.n = n
        self.max_seen = 0.0
        self.k_bits = k_bits

        # ── K-bit advice slicer ───────────────────────────────────────────────
        if self.k_bits > 0:
            num_intervals = 2 ** self.k_bits
            ratio = self.global_M / self.global_m
            self.m = self.global_m * (ratio ** (advice_index / num_intervals))
            self.M = self.global_m * (ratio ** ((advice_index + 1) / num_intervals))
        else:
            self.m = self.global_m
            self.M = self.global_M

        # ── Competitive ratio ─────────────────────────────────────────────────
        # c = ln(M/m) + 1  from El-Yaniv (2001) randomized one-way search
        ratio = self.M / self.m
        self.c = math.log(ratio) + 1.0 if ratio > 1.0 + 1e-9 else 1.0 + 1e-9

        # ── Tier construction ─────────────────────────────────────────────────
        # Divide [m, M] into n_tiers geometrically equal intervals.
        # Direct partitioning: tier_i = m * (M/m)^(i/n_tiers)
        # This avoids the alpha = c/(c-1) problem which breaks for small M/m.
        n_tiers = max(2, int(math.ceil(math.log2(ratio) * 10))) if ratio > 1.0 else 2
        self.tiers = [
            self.m * (ratio ** (i / n_tiers))
            for i in range(1, n_tiers + 1)
        ]

        # ── The randomized die roll (El-Yaniv) ───────────────────────────────
        # Pick ONE tier UNIFORMLY at random — each has probability 1/n_tiers.
        # This is the correct El-Yaniv randomization, not weighted by 1/c.
        self.chosen_tier = random.choice(self.tiers)

    def trade(self, current_price, day_index, date_str):
        action = "HOLD"
        trade_amt = 0.0

        if current_price > self.max_seen:
            self.max_seen = current_price

        # Last day: forced liquidation
        if self.n is not None and day_index == self.n:
            if self.shares > 1e-9:
                trade_amt = self.shares
                action = "SELL_ALL (Last Day)"

        # Sell everything when price reaches the randomly chosen tier
        elif self.shares > 1e-9 and current_price >= self.chosen_tier:
            trade_amt = self.shares
            action = f"SELL ALL (random tier=${self.chosen_tier:.2f})"

        if trade_amt > 1e-9:
            self.shares -= trade_amt
            self.cash += trade_amt * current_price

        self.trades.append({
            "Day": day_index,
            "Date": date_str,
            "Price": current_price,
            "Action": action,
            "Sold": trade_amt,
            "Cash": self.cash,
            "Shares": self.shares
        })