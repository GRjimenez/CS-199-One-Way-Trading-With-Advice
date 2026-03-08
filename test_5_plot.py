import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv(r"C:\Users\Daniel Yap\Desktop\CS199\CS-199-One-Way-Trading-With-Advice\test_5_stable.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort just in case
df = df.sort_values("Date")

# Plot Close Price
plt.figure()
plt.plot(df["Date"], df["Close/Last"])
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Stock Close Price Over Time")
plt.show()