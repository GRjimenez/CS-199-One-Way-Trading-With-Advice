import math

class OneBitTwoSearchTrader:
    def __init__(self, m, M, advice_bit, initial_shares=100.0):
        self.m = float(m)
        self.M = float(M)
        self.advice_bit = advice_bit
        
        self.initial_shares = initial_shares
        self.shares = initial_shares
        self.cash = 0.0
        self.trades = []
        
        # 2-Search means we split our budget into exactly 2 transactions
        self.units_to_sell = 2
        self.shares_per_unit = initial_shares / 2.0
        
        # The core threshold from the Clemente et al. paper: sqrt(M * m)
        self.threshold = math.sqrt(self.M * self.m)
        
        # Calculate the 2 Reservation Prices based on the Advice Bit
        if self.advice_bit == 0:
            # Oracle predicts WEAK market (Max price will be below threshold)
            self.r_prices = [
                math.sqrt(self.m * self.threshold), 
                self.threshold
            ]
        else:
            # Oracle predicts STRONG market (Max price will be above threshold)
            self.r_prices = [
                self.threshold, 
                math.sqrt(self.threshold * self.M)
            ]
            
        self.current_target_index = 0

    def trade(self, current_price, day_index, n_days, date_str):
        action = "HOLD"
        trade_amt = 0.0
        
        # Rule 1: The Final Day Dump
        if day_index == n_days:
            if self.shares > 0:
                trade_amt = self.shares
                action = "SELL_ALL (Last Day)"
                
        # Rule 2: Check against Reservation Prices
        elif self.current_target_index < self.units_to_sell:
            target_price = self.r_prices[self.current_target_index]
            
            # If the current price crosses our hard target, sell exactly 1 unit
            if current_price >= target_price:
                trade_amt = self.shares_per_unit
                action = f"SELL Unit {self.current_target_index + 1} (Target: ${target_price:.2f})"
                
                # Move onto the next price target
                self.current_target_index += 1
                
        # Execute the trade if one was triggered
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