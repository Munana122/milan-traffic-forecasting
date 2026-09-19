# Milan Internet Traffic Forecasting

Comparative analysis of SARIMA, LSTM, and XGBoost for one-step-ahead mobile
network traffic forecasting using the Telecom Italia Big Data Challenge dataset
(Milan, Nov 2013 – Jan 2014).

## Repository structure

```
src/
  loader.py       — one-time ingestion pipeline (raw .txt → Parquet)
  data.py         — canonical load function used by all later scripts
  eda.py          — Section 2: EDA figures (distribution, time series, ACF, decomposition)
  eda_stats.py    — Section 2: prints all summary statistics
  tuning.py       — Section 4: iterative hyperparameter tuning experiments
  experiments.py  — Section 4: final model training and evaluation
data/
  raw/            — staging area (files deleted after processing, not tracked)
  processed/      — milan_internet_traffic.parquet (not tracked, rebuild with loader.py)
figures/
  traffic_distribution.png
  first_two_weeks.png
  acf_top_square.png
  decomposition_top_square.png
  section4/       — per-square prediction plots
notes/
  full_report.md          — complete assembled report
  section1_report.md
  section2_report.md
  section3_report.md
  section4_report.md
  loader_memory_benchmark.md
results/
  section4_results.csv    — MAE / MAPE / RMSE for all models × all squares
  tuning_log.csv          — hyperparameter tuning experiment log
```

## Setup

### 1. Python version
Python 3.10+ recommended (tested on 3.14).

### 2. Install dependencies
```bash
pip install pandas pyarrow statsmodels matplotlib xgboost scikit-learn torch
```

Or install from the requirements file:
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the 62 daily `.txt` files from the
[Telecom Italia Big Data Challenge](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
and place them in folders at the project root:
```
dataverse_files/          ← November 2013 files
dataverse_files (1)/      ← December 2013 files
dataverse_files (2)/      ← Dec 31 2013 + Jan 1 2014
```

## Running the pipeline

### Step 1 — Build the Parquet file (run once)
```bash
python src/loader.py
```
Reads all 62 raw files, aggregates internet traffic per (square_id, time_interval),
deletes each raw file after processing, and writes:
`data/processed/milan_internet_traffic.parquet` (409 MB, ~89 M rows).

### Step 2 — Exploratory analysis
```bash
python src/eda.py
```
Saves four figures to `figures/`. Prints top-3 square IDs and ADF result.

### Step 3 — Hyperparameter tuning
```bash
python src/tuning.py
```
Runs iterative tuning experiments on square 5161 using a validation split
(Dec 9–15). Saves `results/tuning_log.csv`.

### Step 4 — Final experiments
```bash
python src/experiments.py
```
Trains and evaluates all three models on the top-3 squares for Dec 16–22.
Saves `results/section4_results.csv` and three prediction plots to
`figures/section4/`.

## Key results

| Model | Avg MAE | Avg RMSE | Avg train (s) |
|---|---|---|---|
| SARIMA | 72.8 | 104.9 | 7.1 |
| LSTM | 91.8 | 132.8 | 68.7 |
| XGBoost | 75.1 | 111.0 | 1.5 |

Top-3 squares by total traffic: **5161**, 5059, 5259.

## Hardware
All experiments run on CPU (Intel, Windows 11). No GPU required.
