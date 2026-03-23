# maximum_forecast.py
from scipy.optimize import brentq
import numpy as np

class BaseThreatTrader:
    """
    Threat-based trader with λ-advice + safety portion.
    """

    def __init__(self, n, m, M, initial_shares=100.0, lambda_val=0.5, ml_predictor=None):
        self.n = int(n)
        self.m = float(m)
        self.M = float(M)
        self.initial_shares = initial_shares
        self.shares = initial_shares
        self.cash = 0.0
        self.trades = []

        # λ-advice portion
        self.lambda_val = lambda_val
        self.advice_budget = lambda_val * initial_shares
        self.safety_budget = (1 - lambda_val) * initial_shares
        self.ml_predictor = ml_predictor
        self.max_seen = 0.0

        # Compute threat-based constant c
        try:
            self.c = brentq(
                lambda r: r - self.n * (max(0, 1 - (self.m * (r - 1)) / (self.M - self.m)))**(1.0/self.n),
                1.000001,
                self.M / self.m
            )
        except Exception:
            self.c = self.M / self.m

    def trade(self, current_price, day_index, date_str, past_window=None, advice=None, lambda_val=None):
        action = "HOLD"
        total_sold = 0.0

        # Update max seen
        if current_price > self.max_seen:
            self.max_seen = current_price

        # --- Advice portion ---
        if self.lambda_val > 0 and self.advice_budget > 0 and self.ml_predictor and past_window is not None:
            predicted_max = self.ml_predictor.predict(past_window)
            threshold = 0.95 * predicted_max
            if current_price >= threshold:
                sold = self.advice_budget
                self.advice_budget -= sold
                self.shares -= sold
                self.cash += sold * current_price
                action = "SELL_ADVICE"
                total_sold += sold

        # --- Safety portion ---
        if self.safety_budget > 0:
            if day_index == self.n:
                sold = self.safety_budget
                if action == "HOLD":
                    action = "SELL_ALL (Last Day)"
                self.safety_budget -= sold
                self.shares -= sold
                self.cash += sold * current_price
                total_sold += sold
            elif current_price > self.max_seen:
                numerator = (current_price * self.initial_shares) - self.c * (self.cash + self.shares * self.m)
                denominator = self.c * (current_price - self.m)
                if denominator > 0:
                    s_i = max(0.0, min(numerator / denominator, self.safety_budget))
                    if s_i > 1e-4 and action == "HOLD":
                        sold = s_i
                        self.safety_budget -= sold
                        self.shares -= sold
                        self.cash += sold * current_price
                        total_sold += sold
                        action = "SELL_SAFETY_NEW_MAX"

        self.trades.append({
            "Date": date_str,
            "Price": current_price,
            "Action": action,
            "Sold": total_sold,
            "Cash": self.cash,
            "Shares": self.shares
        })