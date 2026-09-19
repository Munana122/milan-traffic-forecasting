"""
Section 2 -- Exploratory Analysis
Reads the aggregated Parquet file produced in Section 1
(columns: square_id, time_interval, internet_traffic) and produces:
    1. Distribution of total traffic across all squares
    2. Top-3 squares by total traffic
    3. First-two-weeks time series for top-3 + square 4159 + square 4556
    4. Two deeper analyses on the highest-traffic square:
       ACF, seasonal decomposition, and ADF stationarity test
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

PARQUET_PATH = "data/processed/milan_internet_traffic.parquet"
FIG_DIR = "figures"
SQUARE_4159 = 4159
SQUARE_4556 = 4556

os.makedirs(FIG_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["timestamp"] = pd.to_datetime(df["time_interval"], unit="ms") + pd.Timedelta(hours=1)
    return df


def total_traffic_distribution(df: pd.DataFrame) -> pd.Series:
    totals = df.groupby("square_id")["internet_traffic"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(totals, bins=100)
    axes[0].set_title("Distribution of total Internet traffic per square")
    axes[0].set_xlabel("Total Internet traffic (two-month sum)")
    axes[0].set_ylabel("Number of squares")

    sorted_vals = totals.sort_values()
    cdf = (sorted_vals.rank() / len(sorted_vals)).values
    axes[1].plot(sorted_vals.values, cdf)
    axes[1].set_title("CDF of total traffic across squares")
    axes[1].set_xlabel("Total Internet traffic")
    axes[1].set_ylabel("Cumulative fraction of squares")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/traffic_distribution.png", dpi=150)
    plt.close()

    print("Summary statistics of total traffic per square:")
    print(totals.describe())
    return totals


def top3_squares(totals: pd.Series) -> list:
    top3 = totals.nlargest(3)
    print("\nTop 3 squares by total Internet traffic:")
    print(top3)
    return list(top3.index)


def first_two_weeks_plot(df: pd.DataFrame, squares_to_plot: list):
    start = df["timestamp"].min()
    end = start + pd.Timedelta(days=14)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

    fig, ax = plt.subplots(figsize=(14, 5))
    for sq in squares_to_plot:
        sub = window[window["square_id"] == sq].sort_values("timestamp")
        ax.plot(sub["timestamp"], sub["internet_traffic"], label=f"Square {sq}")
    ax.set_title("Internet traffic: first two weeks")
    ax.set_xlabel("Time")
    ax.set_ylabel("Internet traffic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/first_two_weeks.png", dpi=150)
    plt.close()


def deep_dive_top_square(df: pd.DataFrame, top_square: int):
    sub = df[df["square_id"] == top_square].sort_values("timestamp").set_index("timestamp")
    series = sub["internet_traffic"].asfreq("10min")
    series = series.interpolate()

    # --- Analysis 1: ACF (up to 3 days of lags) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(series, lags=144 * 3, ax=ax)
    ax.set_title(f"Autocorrelation -- Square {top_square}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/acf_top_square.png", dpi=150)
    plt.close()

    # --- Analysis 2: seasonal decomposition (daily period = 144 intervals) ---
    decomposition = seasonal_decompose(series, period=144, model="additive")
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decomposition_top_square.png", dpi=150)
    plt.close()

    # --- Stationarity test ---
    adf_stat, adf_pvalue, *_ = adfuller(series.dropna())
    print(f"\nAugmented Dickey-Fuller test on square {top_square}:")
    print(f"  ADF statistic = {adf_stat:.4f}, p-value = {adf_pvalue:.4g}")
    print("  (p < 0.05 suggests the series is stationary)")


if __name__ == "__main__":
    df = load_data()
    totals = total_traffic_distribution(df)
    top3 = top3_squares(totals)

    squares_to_plot = top3 + [SQUARE_4159, SQUARE_4556]
    first_two_weeks_plot(df, squares_to_plot)

    highest_square = top3[0]
    deep_dive_top_square(df, highest_square)

    print(f"\nDone. Figures saved to '{FIG_DIR}/'.")
