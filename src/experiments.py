"""
Section 4 -- Forecasting Experiments
Three models: SARIMA (ARIMA + Fourier terms), LSTM (PyTorch), XGBoost (lag features).
One-step-ahead forecasting for the top-3 highest-traffic squares (5161, 5059, 5259),
evaluated over Dec 16-22, 2013.

Outputs:
  figures/section4/square_{sq}_all_models.png   — 3 overlay summary plots
  figures/section4/square_{sq}_{model}.png      — 9 individual actual-vs-predicted plots
  figures/section4/worst_period.png             — zoomed failure analysis plot
  results/section4_results.csv
"""

import os
import time
import platform
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

PARQUET_PATH = "data/processed/milan_internet_traffic.parquet"
FIG_DIR      = "figures/section4"
RESULTS_DIR  = "results"

TOP3_SQUARES = [5161, 5059, 5259]

EVAL_START   = pd.Timestamp("2013-12-16")
EVAL_END     = pd.Timestamp("2013-12-23")   # exclusive

SEQ_LEN      = 144    # one day of 10-min history (LSTM) — tuning Round 1 best
FOURIER_K    = 6      # sine/cosine pairs for daily seasonality (SARIMA)
SARIMA_ORDER = (2, 0, 2)   # d=0: ADF confirmed stationarity; tuning grid best
LSTM_EPOCHS  = 20     # tuning Round 3 best
LSTM_HIDDEN  = 64     # tuning Round 2 best
LSTM_LR      = 1e-3   # tuning Round 3 best
LSTM_BATCH   = 128

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def log_hardware():
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        ram_str = f"{ram_gb:.1f} GB RAM"
    except ImportError:
        ram_str = "RAM unknown (pip install psutil)"
    device = "CPU (no CUDA detected)" if not torch.cuda.is_available() else f"GPU: {torch.cuda.get_device_name(0)}"
    print("=== Hardware ===")
    print(f"  OS       : {platform.system()} {platform.release()}")
    print(f"  CPU      : {platform.processor()}")
    print(f"  Memory   : {ram_str}")
    print(f"  PyTorch  : {device}")
    print(f"  LSTM ran on: CPU")
    print()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_series(square_id: int) -> pd.Series:
    df = pd.read_parquet(PARQUET_PATH, columns=["square_id", "time_interval", "internet_traffic"])
    df = df[df["square_id"] == square_id]
    df["timestamp"] = pd.to_datetime(df["time_interval"], unit="ms") + pd.Timedelta(hours=1)
    series = df.set_index("timestamp")["internet_traffic"].sort_index()
    series = series.asfreq("10min").interpolate()
    return series


def train_eval_split(series: pd.Series):
    train = series[series.index < EVAL_START]
    eval_ = series[(series.index >= EVAL_START) & (series.index < EVAL_END)]
    return train, eval_


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": round(mae, 4), "MAPE": round(mape, 4), "RMSE": round(rmse, 4)}


def fourier_features(index: pd.DatetimeIndex, period=144, K=FOURIER_K) -> pd.DataFrame:
    t = np.arange(len(index))
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------------------
# Model 1: SARIMA with Fourier regressors
# ---------------------------------------------------------------------------

def run_sarima(train: pd.Series, eval_: pd.Series):
    full_index  = train.index.append(eval_.index)
    exog_full   = fourier_features(full_index)
    exog_train  = exog_full.iloc[:len(train)]
    exog_eval   = exog_full.iloc[len(train):]

    t0 = time.perf_counter()
    fitted = sm.tsa.statespace.SARIMAX(
        train, exog=exog_train, order=SARIMA_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = []
    state = fitted
    for i in range(len(eval_)):
        fc = state.forecast(1, exog=exog_eval.iloc[[i]]).iloc[0]
        preds.append(fc)
        state = state.append([eval_.iloc[i]], exog=exog_eval.iloc[[i]], refit=False)
    inference_time = time.perf_counter() - t0

    return np.array(preds), train_time, inference_time


# ---------------------------------------------------------------------------
# Model 2: LSTM (PyTorch)
# ---------------------------------------------------------------------------

class _SeqDataset(Dataset):
    def __init__(self, values: np.ndarray, seq_len: int):
        self.v, self.seq_len = values, seq_len

    def __len__(self):
        return len(self.v) - self.seq_len

    def __getitem__(self, i):
        x = torch.tensor(self.v[i: i + self.seq_len], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.v[i + self.seq_len],    dtype=torch.float32)
        return x, y


class _LSTMModel(nn.Module):
    def __init__(self, hidden=LSTM_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def run_lstm(train: pd.Series, eval_: pd.Series):
    scaler   = MinMaxScaler()
    train_sc = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()

    loader  = DataLoader(_SeqDataset(train_sc, SEQ_LEN), batch_size=LSTM_BATCH, shuffle=True)
    model   = _LSTMModel()
    opt     = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()
    model.train()
    for ep in range(LSTM_EPOCHS):
        ep_loss = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(x)
        print(f"  LSTM epoch {ep+1:02d}/{LSTM_EPOCHS}  loss={ep_loss/len(loader.dataset):.5f}")
    train_time = time.perf_counter() - t0

    full_sc = scaler.transform(
        np.concatenate([train.values, eval_.values]).reshape(-1, 1)
    ).flatten()
    n = len(train)

    model.eval()
    t0 = time.perf_counter()
    preds_sc = []
    with torch.no_grad():
        for i in range(len(eval_)):
            window = full_sc[n + i - SEQ_LEN: n + i]
            x = torch.tensor(window, dtype=torch.float32).view(1, SEQ_LEN, 1)
            preds_sc.append(model(x).item())
    inference_time = time.perf_counter() - t0

    preds = scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()
    return preds, train_time, inference_time


# ---------------------------------------------------------------------------
# Model 3: XGBoost on lag + calendar features
# ---------------------------------------------------------------------------

def _make_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    # recent lags (1-12: last 2 hours) + daily lags (144, 288) + weekly lag (1008)
    # weekly lag included: tuning Round 3 showed consistent improvement on validation
    for lag in list(range(1, 13)) + [144, 288, 1008]:
        df[f"lag_{lag}"] = series.shift(lag)
    df["hour"]      = series.index.hour
    df["dayofweek"] = series.index.dayofweek
    df["interval"]  = (series.index.hour * 60 + series.index.minute) // 10
    return df.dropna()


def run_xgboost(train: pd.Series, eval_: pd.Series):
    feat       = _make_features(pd.concat([train, eval_]))
    train_feat = feat[feat.index < EVAL_START]
    eval_feat  = feat[(feat.index >= EVAL_START) & (feat.index < EVAL_END)]

    X_tr, y_tr = train_feat.drop(columns="y"), train_feat["y"]
    X_ev, y_ev = eval_feat.drop(columns="y"),  eval_feat["y"]

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=0,
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_ev)
    inference_time = time.perf_counter() - t0

    return preds, train_time, inference_time, eval_feat.index


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"SARIMA": "steelblue", "LSTM": "tomato", "XGBoost": "seagreen"}


def _fmt_ax(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())


def save_overlay_plot(eval_index, y_true, preds_dict, square_id, path):
    """One plot with all models overlaid — summary figure."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(eval_index, y_true, label="Actual", color="black", lw=1.5)
    for name, preds in preds_dict.items():
        ax.plot(eval_index[:len(preds)], preds, label=name,
                color=COLORS.get(name), alpha=0.85, lw=1.2)
    _fmt_ax(ax)
    ax.set_title(f"Square {square_id} — Dec 16–22: all models")
    ax.set_ylabel("Internet traffic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_individual_plots(eval_index, y_true, preds_dict, square_id, fig_dir):
    """9 individual actual-vs-predicted plots (one per model per square)."""
    for name, preds in preds_dict.items():
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(eval_index, y_true, label="Actual", color="black", lw=1.5)
        ax.plot(eval_index[:len(preds)], preds, label=name,
                color=COLORS.get(name), alpha=0.9, lw=1.2)
        _fmt_ax(ax)
        ax.set_title(f"Square {square_id} — Dec 16–22: {name}")
        ax.set_ylabel("Internet traffic")
        ax.legend()
        plt.tight_layout()
        path = f"{fig_dir}/square_{square_id}_{name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


def save_worst_period_plot(all_preds: dict, fig_dir: str):
    """
    Find the 24-hour window with the highest mean absolute error across all
    models for square 5161, then plot a zoomed actual-vs-predicted for that window.
    """
    eval_index = all_preds["index"]
    y_true     = all_preds["actual"]

    # compute per-step mean absolute error across all three models
    errors = np.mean([
        np.abs(y_true - all_preds["SARIMA"]),
        np.abs(y_true - all_preds["LSTM"]),
        np.abs(y_true - all_preds["XGBoost"]),
    ], axis=0)

    # find 24-hour window (144 steps) with highest mean error
    window = 144
    best_start = int(np.argmax(
        [errors[i:i+window].mean() for i in range(len(errors) - window)]
    ))
    sl = slice(best_start, best_start + window)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eval_index[sl], y_true[sl], label="Actual", color="black", lw=2)
    for name in ["SARIMA", "LSTM", "XGBoost"]:
        ax.plot(eval_index[sl], all_preds[name][sl], label=name,
                color=COLORS[name], alpha=0.85, lw=1.3)
    _fmt_ax(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=20)
    ax.set_title(f"Square 5161 — worst 24-hour window (all models)\n"
                 f"Period: {eval_index[best_start].strftime('%Y-%m-%d %H:%M')} – "
                 f"{eval_index[best_start+window-1].strftime('%H:%M')}")
    ax.set_ylabel("Internet traffic")
    ax.legend()
    plt.tight_layout()
    path = f"{fig_dir}/worst_period_square_5161.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    print(f"  Worst window starts: {eval_index[best_start]}")
    print(f"  Mean error in window: {errors[sl].mean():.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log_hardware()

    records   = []
    sq5161_preds = {}   # store for worst-period analysis

    for sq in TOP3_SQUARES:
        print(f"\n{'='*50}\nSquare {sq}\n{'='*50}")
        series       = load_series(sq)
        train, eval_ = train_eval_split(series)
        print(f"  Train: {len(train):,} steps  |  Eval: {len(eval_):,} steps")
        preds_dict   = {}

        # --- SARIMA ---
        print("Running SARIMA...")
        s_preds, s_tr, s_inf = run_sarima(train, eval_)
        preds_dict["SARIMA"] = s_preds
        records.append({"square_id": sq, "model": "SARIMA",
                        **compute_metrics(eval_.values, s_preds),
                        "train_s": round(s_tr, 2), "inference_s": round(s_inf, 2)})
        print(f"  SARIMA done  train={s_tr:.1f}s  inference={s_inf:.1f}s")

        # --- LSTM ---
        print("Running LSTM...")
        l_preds, l_tr, l_inf = run_lstm(train, eval_)
        preds_dict["LSTM"] = l_preds
        records.append({"square_id": sq, "model": "LSTM",
                        **compute_metrics(eval_.values, l_preds),
                        "train_s": round(l_tr, 2), "inference_s": round(l_inf, 2)})
        print(f"  LSTM done  train={l_tr:.1f}s  inference={l_inf:.1f}s")

        # --- XGBoost ---
        print("Running XGBoost...")
        x_preds, x_tr, x_inf, x_idx = run_xgboost(train, eval_)
        preds_dict["XGBoost"] = x_preds
        records.append({"square_id": sq, "model": "XGBoost",
                        **compute_metrics(eval_.loc[x_idx].values, x_preds),
                        "train_s": round(x_tr, 2), "inference_s": round(x_inf, 2)})
        print(f"  XGBoost done  train={x_tr:.1f}s  inference={x_inf:.1f}s")

        # overlay summary plot
        save_overlay_plot(eval_.index, eval_.values, preds_dict, sq,
                          f"{FIG_DIR}/square_{sq}_all_models.png")
        print(f"  Saved {FIG_DIR}/square_{sq}_all_models.png")

        # 9 individual plots
        save_individual_plots(eval_.index, eval_.values, preds_dict, sq, FIG_DIR)

        # store square 5161 predictions for failure analysis
        if sq == 5161:
            sq5161_preds = {
                "index":   eval_.index,
                "actual":  eval_.values,
                "SARIMA":  s_preds,
                "LSTM":    l_preds,
                "XGBoost": x_preds,
            }

    # worst-period failure analysis on square 5161
    print("\nGenerating worst-period failure analysis...")
    save_worst_period_plot(sq5161_preds, FIG_DIR)

    results = pd.DataFrame(records)
    results.to_csv(f"{RESULTS_DIR}/section4_results.csv", index=False)

    print("\n\n=== RESULTS ===")
    for sq in TOP3_SQUARES:
        print(f"\nSquare {sq}:")
        sub = results[results["square_id"] == sq][
            ["model", "MAE", "MAPE", "RMSE", "train_s", "inference_s"]]
        print(sub.to_string(index=False))
