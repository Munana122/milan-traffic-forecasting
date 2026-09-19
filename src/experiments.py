"""
Section 4 -- Forecasting Experiments
Three models: SARIMA (ARIMA + Fourier terms), LSTM (PyTorch), XGBoost (lag features).
One-step-ahead forecasting for the top-3 highest-traffic squares (5161, 5059, 5259),
evaluated over Dec 16-22, 2013.

SARIMA note: native seasonal_order with period=144 is computationally prohibitive
in statsmodels. We use dynamic harmonic regression instead: plain ARIMA errors with
Fourier-term exogenous regressors to capture daily seasonality. This is a standard
technique for high-frequency seasonal data (Hyndman & Athanasopoulos, 2021, ch. 11).
"""

import os
import time
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

SEQ_LEN      = 144    # one day of 10-min history (LSTM)
N_LAGS       = 144    # lag features for XGBoost (same window)
FOURIER_K    = 6      # sine/cosine pairs for daily seasonality (SARIMA)
SARIMA_ORDER = (2, 0, 2)   # d=0: ADF confirmed stationarity
LSTM_EPOCHS  = 20
LSTM_HIDDEN  = 64
LSTM_LR      = 1e-3
LSTM_BATCH   = 128

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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
    """Sine/cosine pairs encoding position within the daily cycle."""
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

    # rolling one-step-ahead: append each true observation, no refit
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
    scaler     = MinMaxScaler()
    train_sc   = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()

    loader = DataLoader(_SeqDataset(train_sc, SEQ_LEN), batch_size=LSTM_BATCH, shuffle=True)
    model  = _LSTMModel()
    opt    = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()
    model.train()
    for ep in range(LSTM_EPOCHS):
        ep_loss = sum(
            (lambda loss: (opt.zero_grad(), loss.backward(), opt.step(), loss.item() * len(x))[-1])(loss_fn(model(x), y))
            for x, y in loader
        )
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
    # recent lags (1–12: last 2 hours) + daily lags (144, 288: yesterday, 2 days ago)
    for lag in list(range(1, 13)) + [144, 288]:
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

def save_plot(eval_index, y_true, preds_dict: dict, square_id: int, path: str):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(eval_index, y_true, label="Actual", color="black", lw=1.5)
    colors = {"SARIMA": "steelblue", "LSTM": "tomato", "XGBoost": "seagreen"}
    for name, preds in preds_dict.items():
        ax.plot(eval_index[:len(preds)], preds, label=name,
                color=colors.get(name), alpha=0.85, lw=1.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_title(f"Square {square_id} — Dec 16–22 one-step-ahead forecasts")
    ax.set_ylabel("Internet traffic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    records = []

    for sq in TOP3_SQUARES:
        print(f"\n{'='*50}\nSquare {sq}\n{'='*50}")
        series        = load_series(sq)
        train, eval_  = train_eval_split(series)
        print(f"  Train: {len(train):,} steps  |  Eval: {len(eval_):,} steps")
        preds_dict    = {}

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

        save_plot(eval_.index, eval_.values, preds_dict, sq,
                  f"{FIG_DIR}/square_{sq}.png")

    results = pd.DataFrame(records)
    results.to_csv(f"{RESULTS_DIR}/section4_results.csv", index=False)

    print("\n\n=== RESULTS ===")
    for sq in TOP3_SQUARES:
        print(f"\nSquare {sq}:")
        sub = results[results["square_id"] == sq][["model", "MAE", "MAPE", "RMSE", "train_s", "inference_s"]]
        print(sub.to_string(index=False))
