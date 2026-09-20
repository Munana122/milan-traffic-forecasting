# Milan Internet Traffic Forecasting

Comparing SARIMA, LSTM, and XGBoost for one-step-ahead forecasting of mobile
internet traffic, using the Telecom Italia Big Data Challenge dataset
(Milan, Nov 2013 – Jan 2014).

## Layout

src/
loader.py — one-time ingestion: raw .txt files -> Parquet
data.py — shared loader that everything else imports from
eda.py — Section 2 figures (distribution, time series, ACF, decomposition)
eda_stats.py — prints the numbers behind those figures
tuning.py — hyperparameter search for Section 4
experiments.py — final training + evaluation
data/
raw/ — where you drop the downloaded files; gets emptied as loader.py runs
processed/ — milan_internet_traffic.parquet (not tracked, rebuild it yourself)
figures/
traffic_distribution.png, first_two_weeks.png, acf_top_square.png, decomposition_top_square.png
section4/ — per-square prediction plots
notes/
full_report.md + one report per section, plus loader_memory_benchmark.md
results/
section4_results.csv, tuning_log.csv


## Getting set up

Python 3.10+, I've been running it on 3.14.

```bash
pip install -r requirements.txt
```
or by hand:
```bash
pip install pandas pyarrow statsmodels matplotlib xgboost scikit-learn torch
```

### The dataset

Grab the 62 daily `.txt` files from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV). They download split across a few folders — just leave them named however Dataverse gives them to you, at the project root:

dataverse_files/ # November
dataverse_files (1)/ # December
dataverse_files (2)/ # Dec 31 + Jan 1


## Running it

**1. Build the Parquet file — only need to do this once:**
```bash
python src/loader.py
```
Chews through all 62 files one at a time, keeps only what we actually need (Internet traffic per square per 10-min interval, summed across country codes), deletes each raw file as it goes so you're never holding more than one on disk, and dumps the result to `data/processed/milan_internet_traffic.parquet` (~409 MB, ~89M rows). Takes a while — go make coffee.

**2. EDA:**
```bash
python src/eda.py
```
Spits out the four figures and prints the top-3 squares plus the ADF stationarity result.

**3. Tuning:**
```bash
python src/tuning.py
```
Grid-searches over square 5161 (our highest-traffic square) using Dec 9–15 as a validation week, logs every run to `results/tuning_log.csv`.

**4. The actual experiment:**
```bash
python src/experiments.py
```
Trains all three models on the top-3 squares and forecasts Dec 16–22. Writes the metrics table and the prediction plots.

## Where things stand

| Model   | Avg MAE | Avg RMSE | Avg train time (s) |
|---------|---------|----------|---------------------|
| SARIMA  | 72.8    | 104.9    | 7.1                 |
| LSTM    | 91.8    | 132.8    | 68.7                |
| XGBoost | 75.1    | 111.0    | 1.5                 |

Top-3 squares by total traffic: **5161**, 5059, 5259.

Honestly SARIMA and XGBoost are basically tied, and XGBoost gets there way faster. LSTM is currently the worst of the three despite taking the longest to train, which isn't what I expected going in — need to figure out if that's a tuning issue or just how it is on this data before I write it up as a real finding.

## Hardware

CPU only — Intel, Windows 11. Didn't touch a GPU for any of this.
