import pandas as pd
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_traffic
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

df = load_traffic()
df["timestamp"] = pd.to_datetime(df["time_interval"], unit="ms") + pd.Timedelta(hours=1)

totals = df.groupby("square_id")["internet_traffic"].sum()
print("=== Distribution stats ===")
print(totals.describe())
print(f"Skewness       : {totals.skew():.3f}")
print(f"Squares < mean : {(totals < totals.mean()).sum()}")
print(f"99th pct       : {totals.quantile(0.99):.0f}")
print(f"Squares > 99th : {(totals > totals.quantile(0.99)).sum()}")
print(f"Top square share of total: {totals.max()/totals.sum()*100:.2f}%")

top3 = [5161, 5059, 5259]
special = [4159, 4556]
all_sq = top3 + special

print("\n=== First two weeks per-square stats ===")
start = df["timestamp"].min()
end = start + pd.Timedelta(days=14)
w = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
for sq in all_sq:
    s = w[w["square_id"] == sq]["internet_traffic"]
    print(f"Square {sq}: mean={s.mean():.1f}  std={s.std():.1f}  "
          f"max={s.max():.1f}  min={s.min():.3f}  zeros={( s==0).sum()}")

print("\n=== Weekly pattern check (square 5161) ===")
s5161 = df[df["square_id"] == 5161].set_index("timestamp")["internet_traffic"].sort_index()
s5161 = s5161.asfreq("10min").interpolate()
by_dow = s5161.groupby(s5161.index.dayofweek).mean()
days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
for i, v in by_dow.items():
    print(f"  {days[i]}: {v:.1f}")

print("\n=== Hourly pattern (square 5161, avg across all days) ===")
by_hour = s5161.groupby(s5161.index.hour).mean()
for h, v in by_hour.items():
    print(f"  {h:02d}:00  {v:.1f}")

print("\n=== ADF test (square 5161) ===")
adf_stat, pval, lags, nobs, crit, _ = adfuller(s5161.dropna())
print(f"  ADF statistic : {adf_stat:.4f}")
print(f"  p-value       : {pval:.4g}")
print(f"  Lags used     : {lags}")
print(f"  Critical values: {crit}")

print("\n=== ADF test (square 4159) ===")
s4159 = df[df["square_id"] == 4159].set_index("timestamp")["internet_traffic"].sort_index()
s4159 = s4159.asfreq("10min").interpolate()
adf_stat2, pval2, *_ = adfuller(s4159.dropna())
print(f"  ADF statistic : {adf_stat2:.4f}, p-value: {pval2:.4g}")

print("\n=== ADF test (square 4556) ===")
s4556 = df[df["square_id"] == 4556].set_index("timestamp")["internet_traffic"].sort_index()
s4556 = s4556.asfreq("10min").interpolate()
adf_stat3, pval3, *_ = adfuller(s4556.dropna())
print(f"  ADF statistic : {adf_stat3:.4f}, p-value: {pval3:.4g}")
