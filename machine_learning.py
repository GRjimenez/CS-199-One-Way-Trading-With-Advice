# machine_learning.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class MaxPricePredictor:
    """
    ML predictor for maximum future price over next N days.
    """

    def __init__(self, n_future_days=10, model=None):
        self.n_future = n_future_days
        self.model = model if model else RandomForestRegressor(n_estimators=200, random_state=42)

    def create_features(self, df, price_col='Close/Last'):
        prices = df[price_col].values
        X, y = [], []
        for i in range(len(prices) - self.n_future):
            past_window = prices[i:i + self.n_future]
            X.append(past_window)
            future_max = prices[i + 1:i + self.n_future + 1].max()
            y.append(future_max)
        return np.array(X), np.array(y)

    def fit(self, df, price_col='Close/Last'):
        X, y = self.create_features(df, price_col)
        self.model.fit(X, y)
        return self

    def predict(self, past_window):
        past_window = np.array(past_window).reshape(1, -1)
        return self.model.predict(past_window)[0]