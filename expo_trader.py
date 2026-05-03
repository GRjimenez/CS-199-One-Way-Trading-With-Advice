"""
expo_trader.py
==============
Implements the EXPO (exponential reservation price) one-way selling algorithm,
inspired by the randomized algorithm in Fung (2021) and El-Yaniv et al. (2001).

HOW IT WORKS:
=============
The EXPO algorithm divides the price range [m, M] into geometrically-spaced
reservation price tiers. When the current price crosses a tier boundary, it
sells a fixed fraction (1/c) of remaining shares at that price.

This is the one-way selling adaptation of Fung's Algorithm 1 (Section 3.1),
which uses geometric reservation prices to achieve O(log(M/m)) competitive
ratio — much better than the threat-based algorithm's n-dependent ratio when
M/m is large.

KEY DIFFERENCE FROM THREAT-BASED:
===================================
- Threat-based: time-AWARE — uses n (number of days) to compute c
- EXPO:         time-BLIND — only uses M/m ratio, n is irrelevant to strategy
                (n is only needed to force a last-day dump to end the game)

The competitive ratio is: c = ln(M/m) + 1   (El-Yaniv / Fung)

TIER CONSTRUCTION:
==================
We divide [m, M] into n_tiers geometrically-equal intervals:
    tier_i = m * (M/m)^(i/n_tiers)  for i = 1, 2, ..., n_tiers

n_tiers = ceil(log2(M/m) * 10) ensures enough resolution across any M/m ratio.
When a price crosses a tier boundary, sell (1/c) of current remaining shares.

NOTE ON THE ORIGINAL alpha = c/(c-1) FORMULA:
===============================================
The formula alpha = c/(c-1) from El-Yaniv produces astronomically large steps
when c is close to 1 (i.e. when M/m is small, like typical stock data).
For Apple stock with M/m ~1.05-2.5, alpha ranges from 5 to 20, meaning
m*alpha >> M — zero tiers are ever generated and nothing trades until the
last day. We therefore use direct geometric partitioning of [m, M] instead,
which correctly handles all M/m ratios.
"""

import math
import numpy as np

class ExpoTrader:
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

        # ── 1. K-BIT ADVICE SLICER ──────────────────────────────────────────
        # Same geometric partitioning as KBitThreatTrader — oracle tells us
        # which sub-interval of [m, M] contains the true maximum.
        if self.k_bits > 0:
            num_intervals = 2 ** self.k_bits
            ratio = self.global_M / self.global_m
            self.m = self.global_m * (ratio ** (advice_index / num_intervals))
            self.M = self.global_m * (ratio ** ((advice_index + 1) / num_intervals))
        else:
            self.m = self.global_m
            self.M = self.global_M

        # ── 2. COMPETITIVE RATIO ─────────────────────────────────────────────
        # c = ln(M/m) + 1  — closed form from El-Yaniv et al. (2001) / Fung (2021)
        # This is the optimal competitive ratio for the randomized one-way search.
        ratio = self.M / self.m
        if ratio <= 1.0 + 1e-9:
            # Degenerate interval — c defaults to 1, no tiers needed
            self.c = 1.0 + 1e-9
        else:
            self.c = math.log(ratio) + 1.0

        # ── 3. TIER CONSTRUCTION ─────────────────────────────────────────────
        # Divide [m, M] into geometrically-spaced reservation prices.
        # n_tiers = ceil(log2(M/m) * 10) gives enough resolution for any ratio.
        # tier_i = m * (M/m)^(i/n_tiers)
        # At each tier crossing: sell (1/c) of CURRENT remaining shares.
        n_tiers = max(2, int(math.ceil(math.log2(ratio) * 10))) if ratio > 1.0 else 2
        self.tiers = [
            self.m * (ratio ** (i / n_tiers))
            for i in range(1, n_tiers + 1)
        ]
        self.current_tier_idx = 0

    # ── 4. TRADING LOGIC ─────────────────────────────────────────────────────
    def trade(self, current_price, day_index, date_str):
        action = "HOLD"
        trade_amt = 0.0

        if current_price > self.max_seen:
            self.max_seen = current_price

        # Last day: dump everything (forced end of game)
        if self.n is not None and day_index == self.n:
            if self.shares > 1e-9:
                trade_amt = self.shares
                action = "SELL_ALL (Last Day)"
        else:
            # Count how many tier boundaries the current price has crossed
            tiers_crossed = 0
            while (self.current_tier_idx < len(self.tiers)
                   and current_price >= self.tiers[self.current_tier_idx]):
                tiers_crossed += 1
                self.current_tier_idx += 1

            if tiers_crossed > 0 and self.shares > 1e-9:
                # At each tier: sell (1/c) fraction of CURRENT remaining shares.
                # Compound across multiple tiers crossed in one day.
                remaining = self.shares
                total_to_sell = 0.0
                for _ in range(tiers_crossed):
                    sell_this = remaining / self.c
                    total_to_sell += sell_this
                    remaining -= sell_this

                trade_amt = max(0.0, min(total_to_sell, self.shares))
                if trade_amt > 1e-9:
                    action = f"SELL EXPO ({tiers_crossed} tier(s) crossed)"

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


class RandomizedExpoTrader:
    """
    Randomized EXPO trader — directly implements Fung (2021) Algorithm 1
    adapted for one-way selling.

    Instead of selling at ALL tier crossings (deterministic), this trader
    randomly picks ONE reservation price from the geometric set and sells
    everything when that single price is reached.

    This is time-blind (n not used in strategy) and achieves competitive
    ratio O(log(M/m)) in expectation over the random choice.

    For meaningful results on fixed real data, run this many times and
    average — use run_randomized_expo() in runner.py which does 100 trials.

    Args:
        m, M         : price bounds
        n            : total days (only for last-day dump)
        k_bits       : advice bits (0 = no advice)
        advice_index : which geometric sub-interval (from oracle)
        rng          : numpy random Generator (pass for reproducibility)
        initial_shares: starting share count
    """

    def __init__(self, m, M, initial_shares=100.0, n=None,
                 k_bits=0, advice_index=0, rng=None):
        self.global_m = float(m)
        self.global_M = float(M)
        self.initial_shares = float(initial_shares)
        self.shares = float(initial_shares)
        self.cash = 0.0
        self.trades = []
        self.n = n
        self.max_seen = 0.0
        self.k_bits = k_bits
        self.rng = rng if rng is not None else np.random.default_rng()

        # ── K-bit advice slicer (same as ExpoTrader) ─────────────────────────
        if self.k_bits > 0:
            num_intervals = 2 ** self.k_bits
            ratio = self.global_M / self.global_m
            self.m = self.global_m * (ratio ** (advice_index / num_intervals))
            self.M = self.global_m * (ratio ** ((advice_index + 1) / num_intervals))
        else:
            self.m = self.global_m
            self.M = self.global_M

        # ── Competitive ratio ─────────────────────────────────────────────────
        ratio = self.M / self.m
        self.c = math.log(ratio) + 1.0 if ratio > 1.0 + 1e-9 else 1.0 + 1e-9

        # ── Build tier list then pick ONE at random ───────────────────────────
        # Fung Algorithm 1: choose one reservation price uniformly from the
        # geometric set {m*r, m*r^2, ..., M} where r = (M/m)^(1/n_tiers).
        n_tiers = max(2, int(math.ceil(math.log2(ratio) * 10))) if ratio > 1.0 else 2
        all_tiers = [
            self.m * (ratio ** (i / n_tiers))
            for i in range(1, n_tiers + 1)
        ]

        # Randomly pick one reservation price — this is the randomization step
        self.reservation_price = float(self.rng.choice(all_tiers))
        self.sold = False  # only sell once, at the reservation price

    def trade(self, current_price, day_index, date_str):
        action = "HOLD"
        trade_amt = 0.0

        if current_price > self.max_seen:
            self.max_seen = current_price

        # Last day: dump everything
        if self.n is not None and day_index == self.n:
            if self.shares > 1e-9:
                trade_amt = self.shares
                action = "SELL_ALL (Last Day)"

        # Sell everything when reservation price is reached (once only)
        elif not self.sold and current_price >= self.reservation_price and self.shares > 1e-9:
            trade_amt = self.shares
            action = f"SELL RAND-EXPO (RP=${self.reservation_price:.2f})"
            self.sold = True

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