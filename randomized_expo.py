"""
randomized_expo.py
==================
Implements El-Yaniv et al. (2001) EXPO algorithm for one-way selling.

From Section 3 of the paper:
    "Define EXPO to be a uniform probability mixture over the set {RPP(i)}
     where RPP(i) is the reservation price policy with reservation price m*2^i"

Algorithm:
    1. Divide [m, M] into n_tiers geometrically equal reservation prices
    2. Pick ONE uniformly at random — this is the single sell target
    3. Sell ALL shares when price reaches that tier
    4. If price never reaches it, dump everything on the last day

Key theorem (El-Yaniv Theorem 1):
    Any randomized one-way trading algorithm has a deterministic equivalent
    with the same expected return. So EXPO here is really a randomized SEARCH
    algorithm — the randomization is over which single price to target.

Competitive ratio: c = ln(M/m) + 1  (El-Yaniv Section 3)

IMPORTANT: uses numpy rng (not Python's random module) so that the
Monte Carlo runner can control the seed and get reproducible results.
"""

import math
import numpy as np


class RandomizedExpoTrader:
    def __init__(self, m, M, initial_shares=100.0, n=None, k_bits=0,
                 advice_index=0, rng=None):
        self.global_m = float(m)
        self.global_M = float(M)
        self.initial_shares = float(initial_shares)
        self.shares = float(initial_shares)
        self.cash = 0.0
        self.trades = []
        self.n = n
        self.max_seen = 0.0
        self.k_bits = k_bits

        # Use provided rng or create a new one
        # MUST use numpy rng — not Python's random module —
        # so the Monte Carlo runner can seed it for reproducibility
        self.rng = rng if rng is not None else np.random.default_rng()

        # ── K-bit advice slicer ───────────────────────────────────────────────
        # Oracle tells us which geometric sub-interval of [m, M] contains
        # the true maximum, so we trade on tighter bounds [m', M']
        if self.k_bits > 0:
            num_intervals = 2 ** self.k_bits
            ratio = self.global_M / self.global_m
            self.m = self.global_m * (ratio ** (advice_index / num_intervals))
            self.M = self.global_m * (ratio ** ((advice_index + 1) / num_intervals))
        else:
            self.m = self.global_m
            self.M = self.global_M

        # ── Competitive ratio ─────────────────────────────────────────────────
        # c = ln(M/m) + 1  — El-Yaniv (2001) Section 3, Theorem 3
        ratio = self.M / self.m
        self.c = math.log(ratio) + 1.0 if ratio > 1.0 + 1e-9 else 1.0 + 1e-9

        # ── Tier construction ─────────────────────────────────────────────────
        # Divide [m, M] into n_tiers geometrically equal reservation prices.
        # tier_i = m * (M/m)^(i/n_tiers)  for i = 1, 2, ..., n_tiers
        #
        # n_tiers = ceil(log2(M/m) * 10) gives enough resolution for any ratio.
        # The last tier is always exactly M, so the price always has a chance
        # to reach it (since M = max price in the data by definition).
        #
        # We do NOT use alpha = c/(c-1) from El-Yaniv because that formula
        # produces astronomically large steps for small M/m (Apple stock M/m ~1.3),
        # generating zero usable tiers. Direct partitioning solves this.
        n_tiers = max(2, int(math.ceil(math.log2(ratio) * 10))) if ratio > 1.0 else 2
        self.tiers = [
            self.m * (ratio ** (i / n_tiers))
            for i in range(1, n_tiers + 1)
        ]

        # ── Randomized die roll (El-Yaniv EXPO) ──────────────────────────────
        # Pick ONE tier UNIFORMLY at random.
        # Each tier has probability 1/n_tiers — this is the correct
        # El-Yaniv uniform mixture, not weighted by 1/c.
        self.chosen_tier = float(self.rng.choice(self.tiers))
        self.has_sold = False  # ensure we only sell once

    def trade(self, current_price, day_index, date_str):
        action = "HOLD"
        trade_amt = 0.0

        if current_price > self.max_seen:
            self.max_seen = current_price

        # Last day: forced liquidation of anything remaining
        if self.n is not None and day_index == self.n:
            if self.shares > 1e-9:
                trade_amt = self.shares
                action = "SELL_ALL (Last Day)"

        # Sell everything the moment price reaches the chosen tier (once only)
        elif not self.has_sold and self.shares > 1e-9 and current_price >= self.chosen_tier:
            trade_amt = self.shares
            action = f"SELL ALL (tier=${self.chosen_tier:.2f})"
            self.has_sold = True

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
            "Shares": self.shares,
        })