"""
Hyperparameter tuning experiments for Section 4 (Methodology).
Runs iterative experiments on square 5161 (highest-traffic) only,
using a validation split (last 7 days of training data) to select
hyperparameters before final evaluation on the Dec 16-22 test week.

Experiment log is printed and saved to results/tuning_log.csv.
"""

import os, time, warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import statsmodels.api as sm

PARQUET_PATH = "data/processed/milan_internet_traffic.parquet"
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SQUARE       = 5161
EVAL_START   = pd.Timestamp("2013-12-16")
VAL_START    = pd.Timestamp("2013-12-09")   # 1-week validation window before test
FOURIER_K    = 6


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_series(square_id):
    df = pd.read_parquet(PARQUET_PATH, columns=["square_id","time_interval","internet_traffic"])
    df = df[df["square_id"] == square_id]
    df["timestamp"] = pd.to_datetime(df["time_interval"], unit="ms") + pd.Timedelta(hours=1)
    s = df.set_index("timestamp")["internet_traffic"].sort_index().asfreq("10min").interpolate()
    return s

def splits(series):
    train_full = series[series.index < EVAL_START]
    train_cv   = series[series.index < VAL_START]
    val        = series[(series.index >= VAL_START) & (series.index < EVAL_START)]
    test       = series[(series.index >= EVAL_START) & (series.index < EVAL_START + pd.Timedelta(days=7))]
    return train_full, train_cv, val, test

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    return mae, mape, rmse

def fourier_features(index, period=144, K=FOURIER_K):
    t = np.arange(len(index))
    cols = {}
    for k in range(1, K+1):
        cols[f"sin_{k}"] = np.sin(2*np.pi*k*t/period)
        cols[f"cos_{k}"] = np.cos(2*np.pi*k*t/period)
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------------------
# SARIMA tuning  — grid over ARIMA order (p, q) with d=0
# ---------------------------------------------------------------------------

def sarima_val(train, val, order):
    full_idx   = train.index.append(val.index)
    exog_full  = fourier_features(full_idx)
    exog_train = exog_full.iloc[:len(train)]
    exog_val   = exog_full.iloc[len(train):]
    try:
        fitted = sm.tsa.statespace.SARIMAX(
            train, exog=exog_train, order=order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        preds, state = [], fitted
        for i in range(len(val)):
            fc = state.forecast(1, exog=exog_val.iloc[[i]]).iloc[0]
            preds.append(fc)
            state = state.append([val.iloc[i]], exog=exog_val.iloc[[i]], refit=False)
        return metrics(val.values, np.array(preds))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LSTM tuning
# ---------------------------------------------------------------------------

class SeqDS(Dataset):
    def __init__(self, v, seq_len):
        self.v, self.seq_len = v, seq_len
    def __len__(self): return len(self.v) - self.seq_len
    def __getitem__(self, i):
        x = torch.tensor(self.v[i:i+self.seq_len], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.v[i+self.seq_len],   dtype=torch.float32)
        return x, y

class LSTMModel(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True, dropout=0.1 if layers>1 else 0)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(-1)

def lstm_val(train, val, seq_len, hidden, layers, epochs, lr):
    scaler   = MinMaxScaler()
    tr_sc    = scaler.fit_transform(train.values.reshape(-1,1)).flatten()
    loader   = DataLoader(SeqDS(tr_sc, seq_len), batch_size=128, shuffle=True)
    model    = LSTMModel(hidden, layers)
    opt      = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn  = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    full_sc = scaler.transform(np.concatenate([train.values, val.values]).reshape(-1,1)).flatten()
    n = len(train)
    model.eval()
    preds_sc = []
    with torch.no_grad():
        for i in range(len(val)):
            w = full_sc[n+i-seq_len:n+i]
            preds_sc.append(model(torch.tensor(w, dtype=torch.float32).view(1,seq_len,1)).item())
    preds = scaler.inverse_transform(np.array(preds_sc).reshape(-1,1)).flatten()
    return metrics(val.values, preds)


# ---------------------------------------------------------------------------
# XGBoost tuning
# ---------------------------------------------------------------------------

def make_features(series, lags_recent, lag_daily):
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags_recent+1):
        df[f"lag_{lag}"] = series.shift(lag)
    for lag in lag_daily:
        df[f"lag_{lag}"] = series.shift(lag)
    df["hour"]      = series.index.hour
    df["dayofweek"] = series.index.dayofweek
    df["interval"]  = (series.index.hour*60 + series.index.minute)//10
    return df.dropna()

def xgb_val(train, val, n_estimators, max_depth, lr, lags_recent, lag_daily):
    feat      = make_features(pd.concat([train, val]), lags_recent, lag_daily)
    tr_feat   = feat[feat.index < VAL_START]
    val_feat  = feat[(feat.index >= VAL_START) & (feat.index < EVAL_START)]
    X_tr, y_tr = tr_feat.drop(columns="y"), tr_feat["y"]
    X_v,  y_v  = val_feat.drop(columns="y"), val_feat["y"]
    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                              learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                              n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr)
    return metrics(y_v.values, model.predict(X_v))


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    series = load_series(SQUARE)
    train_full, train_cv, val, test = splits(series)
    records = []

    print(f"Square {SQUARE} | train_cv={len(train_cv)} val={len(val)} test={len(test)}\n")

    # ── SARIMA ──────────────────────────────────────────────────────────────
    print("=== SARIMA grid (p, q) with d=0 ===")
    sarima_grid = [(1,1),(1,2),(2,1),(2,2),(3,2),(2,3)]
    sarima_results = []
    for order in sarima_grid:
        full_order = (order[0], 0, order[1])
        r = sarima_val(train_cv, val, full_order)
        if r:
            mae, mape, rmse = r
            print(f"  ARIMA{full_order}  MAE={mae:.2f}  MAPE={mape:.2f}%  RMSE={rmse:.2f}")
            sarima_results.append((full_order, mae, rmse))
            records.append({"model":"SARIMA","experiment":str(full_order),
                            "MAE":round(mae,4),"MAPE":round(mape,4),"RMSE":round(rmse,4),
                            "split":"val","notes":f"order={full_order}"})
    best_sarima = min(sarima_results, key=lambda x: x[1])[0]
    print(f"  → Best SARIMA order: {best_sarima}\n")

    # ── LSTM ─────────────────────────────────────────────────────────────────
    print("=== LSTM iterative experiments ===")
    # Round 1: vary seq_len
    print("  Round 1: seq_len (hidden=32, layers=1, epochs=15, lr=0.001)")
    for seq_len in [72, 144, 288]:
        mae, mape, rmse = lstm_val(train_cv, val, seq_len, 32, 1, 15, 0.001)
        print(f"    seq_len={seq_len}  MAE={mae:.2f}  MAPE={mape:.2f}%  RMSE={rmse:.2f}")
        records.append({"model":"LSTM","experiment":f"seq={seq_len}_h=32_l=1_ep=15_lr=0.001",
                        "MAE":round(mae,4),"MAPE":round(mape,4),"RMSE":round(rmse,4),
                        "split":"val","notes":"Round 1: vary seq_len"})

    # Round 2: vary hidden size (best seq_len from round 1)
    print("  Round 2: hidden size (seq_len=144, layers=1, epochs=15, lr=0.001)")
    for hidden in [32, 64, 128]:
        mae, mape, rmse = lstm_val(train_cv, val, 144, hidden, 1, 15, 0.001)
        print(f"    hidden={hidden}  MAE={mae:.2f}  MAPE={mape:.2f}%  RMSE={rmse:.2f}")
        records.append({"model":"LSTM","experiment":f"seq=144_h={hidden}_l=1_ep=15_lr=0.001",
                        "MAE":round(mae,4),"MAPE":round(mape,4),"RMSE":round(rmse,4),
                        "split":"val","notes":"Round 2: vary hidden"})

    # Round 3: vary epochs and lr (best hidden from round 2)
    print("  Round 3: epochs + lr (seq_len=144, hidden=64, layers=1)")
    for epochs, lr in [(15, 0.001), (20, 0.001), (20, 0.0005)]:
        mae, mape, rmse = lstm_val(train_cv, val, 144, 64, 1, epochs, lr)
        print(f"    epochs={epochs} lr={lr}  MAE={mae:.2f}  MAPE={mape:.2f}%  RMSE={rmse:.2f}")
        records.append({"model":"LSTM","experiment":f"seq=144_h=64_l=1_ep={epochs}_lr={lr}",
                        "MAE":round(mae,4),"MAPE":round(mape,4),"RMSE":round(rmse,4),
                        "split":"val","notes":"Round 3: vary epochs+lr"})

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("\n=== XGBoost iterative experiments ===")
    # Round 1: vary lag window
    print("  Round 1: recent lag window (n_est=300, depth=5, lr=0.05)")
    for n_recent in [6, 12, 24]:
        mae, mape, rmse = xgb_val(train_cv, val, 300, 5, 0.05, n_recent, [144, 288])
        print(f"    lags_recent={n_recent}  MAE={mae:.2f}  MAPE={mape:.2f}%  RMSE={rmse:.2f}")
        records.append({"model":"XGBoost","experiment":f"lags={n_recent}_est=300_d=5_lr=0.05",
                        "MAE":round(mae,4),"MAPE":round(mape,4),"RMSE":round(rmse,4),
                        "split":"val","notes":"Round 1: vary lag window"})

    # Round 2: vary n_estimators and depth (best lag from round 1)
    print("  Round 2: n_estimators + max_depth (lags_recent=12)")
    for n_est, depth in [(200,4),(300,5),(500,6),(500,8)]:
        mae, mape, rmse = xgb_val(train_cv, val, n_est, depth, 0.05, 12, [144, 288])
        print(f"    n_est={n_est} depth={depth}  MAE={mae:.2f}  MAPE={mape:.2f}%  RMSE={rmse:.2f}")
        records.append({"model":"XGBoost","experiment":f"lags=12_est={n_est}_d={depth}_lr=0.05",
                        "MAE":round(mae,4),"MAPE":round(mape,4),"RMSE":round(rmse,4),
                        "split":"val","notes":"Round 2: vary n_est+depth"})

    # Round 3: add weekly lag (lag 1008)
    print("  Round 3: add weekly lag (lag 1008) with best config")
    for lag_daily in [[144,288],[144,288,1008]]:
        mae, mape, rmse = xgb_val(train_cv, val, 500, 6, 0.05, 12, lag_daily)
        label = "+weekly" if 1008 in lag_daily else "no_weekly"
        print(f"    daily_lags={lag_daily}  MAE={mae:.2f}  MAPE={mape:.2f}%  RMSE={rmse:.2f}")
        records.append({"model":"XGBoost","experiment":f"lags=12_est=500_d=6_{label}",
                        "MAE":round(mae,4),"MAPE":round(mape,4),"RMSE":round(rmse,4),
                        "split":"val","notes":f"Round 3: daily_lags={lag_daily}"})

    df_log = pd.DataFrame(records)
    df_log.to_csv(f"{RESULTS_DIR}/tuning_log.csv", index=False)
    print(f"\nTuning log saved to {RESULTS_DIR}/tuning_log.csv")
    print("\n=== Full tuning log ===")
    print(df_log.to_string(index=False))
