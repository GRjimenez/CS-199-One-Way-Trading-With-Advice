import pandas as pd
import matplotlib.pyplot as plt
from ml_utils import load_and_preprocess

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

def plot_prices(file_path):
    # Load and clean data
    df = load_and_preprocess(file_path)

    # Plot closing price
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close/Last'])

    plt.title("Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_prices(get_path('AppleData_2017.csv'))