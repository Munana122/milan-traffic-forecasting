# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

---

## 1. Introduction

Mobile network traffic forecasting is a core problem in telecommunications
infrastructure management. Accurate short-horizon predictions enable network
operators to dynamically allocate bandwidth, pre-empt congestion, and optimise
energy consumption in base stations (Trinh et al., 2018). As mobile data demand
grows and traffic patterns become increasingly complex, the choice of forecasting
model — and the ability to justify that choice empirically — has direct operational
consequences.

This study investigates the following research question: *How do different
sequential models compare for one-step-ahead mobile network traffic forecasting,
and how does their performance vary across geographical areas with different
traffic characteristics?*

Three architecturally distinct models are implemented and evaluated: SARIMA with
Fourier regressors (a classical statistical approach), LSTM (a deep recurrent
model), and XGBoost on engineered lag features (a feature-based machine learning
approach). The dataset is the Telecom Italia Big Data Challenge (Barlacchi et al.,
2015), covering Milan's 10,000-cell grid at 10-minute resolution over approximately
two months. The objective is not simply to identify the model with the lowest
prediction error, but to investigate how each model's design relates to the
observed data characteristics, and to draw conclusions supported by systematic
experimentation.

---

## 2. Related Work

Mobile and cellular network traffic forecasting has been studied extensively,
with approaches ranging from classical statistical models to deep learning
architectures.

**Statistical baselines.** ARIMA and its seasonal extension SARIMA have long
served as standard baselines. Their interpretability and principled treatment of
trend and seasonality make them attractive for periodic traffic series. However,
they assume linear dynamics and struggle with the nonlinear, bursty behaviour
common in high-traffic urban cells (Box et al., 2015). For high-frequency data
with large seasonal periods, native seasonal ARIMA becomes computationally
prohibitive; dynamic harmonic regression — replacing seasonal structure with
Fourier-term exogenous regressors — is a well-established alternative
(Hyndman & Athanasopoulos, 2021, §11.1).

**Deep sequential models.** LSTM networks became the dominant approach for
traffic forecasting after demonstrating the ability to learn long-range temporal
dependencies without manual feature engineering. A study using this exact
Telecom Italia Milan dataset compared LSTM and GRU architectures, applying
K-Means clustering to group cells by activity level and grid search for
hyperparameter tuning, and found LSTM outperformed GRU overall while capturing
both daily and two-month seasonality patterns (Qiu et al., 2023). A broader
survey confirms LSTM's consistent advantage over plain RNNs, while noting that
training cost scales with parameter count — a relevant trade-off in empirical
comparison (Shi et al., 2021).

**Hybrid approaches.** Wang et al. (2022) demonstrated that pairing
double-seasonal ARIMA with LSTM — where ARIMA captures the linear seasonal
structure and LSTM models the nonlinear residuals — outperformed either component
alone. This implies that cellular traffic contains both a strong periodic/linear
component and a nonlinear residual that purely statistical models leave
unexplained.

**Feature-based machine learning.** Gradient boosting methods reframe forecasting
as supervised regression over engineered lag and calendar features. They train
faster than deep models, require no sequence architecture, and provide
feature-importance interpretability. Chen & Guestrin (2016) demonstrated XGBoost's
competitive accuracy on tabular regression benchmarks, and it has since been widely
applied to time-series forecasting problems where domain knowledge can be encoded
as lag features.

**Implications for this study.** The literature motivates three distinct model
classes: a statistical baseline (SARIMA), a deep sequential model (LSTM), and a
feature-based ML model (XGBoost). The Milan-specific findings of Qiu et al. (2023)
provide direct grounding for the LSTM choice, while the hybrid results of Wang et
al. (2022) suggest that the linear/nonlinear decomposition of traffic is a
meaningful axis of comparison.

---

## 3. Dataset and Data Preparation

### 3.1 Dataset

The raw dataset is the Telecom Italia Big Data Challenge (Barlacchi et al., 2015),
covering Milan's 10,000-cell grid at 10-minute resolution from 1 November 2013 to
1 January 2014 — 62 daily tab-separated files totalling 20.8 GB on disk. Each row
records activity for one grid square, one time interval, and one country code across
eight columns: square_id, time_interval, country_code, sms_in, sms_out, call_in,
call_out, and internet_traffic. Only three columns are needed for this study.

### 3.2 Memory management strategy

Loading all 62 files naively — all eight columns, default int64/float64 dtypes, no
early aggregation — would require approximately 62 × 349 MB ≈ 21 GB in memory
simultaneously. Four targeted optimisations were applied:

- **usecols**: `pd.read_csv(..., usecols=[...])` discards the five SMS and call
  columns at parse time, before they are allocated in memory. Pandas' CSV engine
  skips these fields entirely rather than parsing and dropping them (McKinney, 2022).
- **Dtype downcasting**: square_id stored as int32 (max value 10,000) and
  internet_traffic as float32, halving per-cell cost relative to pandas' default
  int64/float64.
- **Early aggregation**: each raw file contains one row per (square_id,
  time_interval, country_code) triplet. Summing internet_traffic across country
  codes immediately after loading collapses ~4.8 M rows to ~1.4 M rows (70%
  reduction) before the DataFrame is appended to the accumulator list.
- **Process-then-discard**: each raw file is copied to a staging folder, processed,
  and deleted before the next file is read. At no point does more than one 322 MB
  raw file exist on disk simultaneously.

### 3.3 Evidence of memory reduction

Measured with Python's `tracemalloc` on one representative file
(`sms-call-internet-mi-2013-11-01.txt`, 322 MB on disk):

| Method | Peak memory | Rows |
|---|---|---|
| Naive (all columns, default dtypes) | 348.95 MB | 4,842,625 |
| Optimised (usecols + int32/float32 + groupby) | 355.99 MB | 1,439,982 |
| **Retained after aggregation** | **23 MB** | **1,439,982** |

The peak figures are similar because `tracemalloc` captures the high-water mark
during loading, before aggregation frees memory. The meaningful difference is in
the retained DataFrame: 23 MB vs. the naive 349 MB peak — a 93% reduction in
working-set size. Across the full 62-file pipeline, the whole-loop peak was
2,857 MB, driven by the accumulator list growing before `pd.concat`.

### 3.4 Parquet output

After concatenation, the combined DataFrame was written to a single Parquet file
using Snappy compression:

| | Size |
|---|---|
| Raw `.txt` files (62 days) | 20.805 GB |
| `milan_internet_traffic.parquet` | 408.79 MB |
| Reduction | **98.0% — 50.9× smaller** |

The Parquet file reloads in 2.4 s via `pd.read_parquet` (89,245,318 rows × 3
columns). All analysis in subsequent sections reads exclusively from this file.

---

## 4. Exploratory Analysis

### 4.1 Distribution of total traffic across squares

Figure 1 shows the histogram and CDF of total Internet traffic summed over the
full two-month period across all 10,000 grid squares.

| Statistic | Value |
|---|---|
| Mean | 555,289 |
| Median | 277,871 |
| Std deviation | 892,596 |
| Skewness | 4.27 |
| 99th percentile | 4,698,732 |
| Maximum (square 5161) | 12,740,060 |

The distribution is strongly right-skewed (skewness = 4.27): 73% of squares fall
below the mean, and the top 1% (100 squares) account for a disproportionate share
of total traffic. This spatial heterogeneity is consistent with an urban area where
a small number of high-density commercial or transport hubs dominate network load.
It motivates evaluating models across squares with different traffic levels, as
performance may vary substantially between low- and high-traffic areas.

### 4.2 Top-3 squares and time-series comparison

| Rank | Square ID | Total traffic |
|---|---|---|
| 1 | **5161** | 12,740,060 |
| 2 | 5059 | 11,170,854 |
| 3 | 5259 | 10,485,780 |

Figure 2 shows Internet traffic at 10-minute resolution for the first 14 days
(1–14 November 2013) for the top-3 squares plus squares 4159 and 4556.

| Square | Mean | Std | Max |
|---|---|---|---|
| 5161 | 1483.5 | 1312.8 | 8044.1 |
| 5059 | 1330.6 | 952.4 | 4182.8 |
| 5259 | 1293.5 | 1136.7 | 4262.9 |
| 4159 | 311.3 | 189.3 | 851.1 |
| 4556 | 592.7 | 255.6 | 1868.7 |

All five squares exhibit a clear daily cycle peaking around 15:00–17:00 local
time and troughing at 03:00–05:00. No square records zero traffic at any interval.
The top-3 squares operate at 4–5× the level of squares 4159 and 4556. Square 5161
shows notably higher variability (std = 1312.8) and a maximum of 8044.1 — roughly
5× its mean — suggesting proximity to a venue or transport hub generating
occasional large spikes. These anomalous bursts are localised to square 5161 and
do not appear in other squares simultaneously, ruling out network-wide effects.

A weekly pattern is also visible: Saturday and Sunday show a later morning
ramp-up and broader afternoon peak compared to weekdays (Saturday mean 1849.5,
Sunday 1704.7 vs. weekday average ~1280 for square 5161).

### 4.3 Autocorrelation analysis (square 5161)

Figure 3 shows the ACF of square 5161's full series up to 432 lags (3 days).

Three features are evident:

1. **Strong short-range autocorrelation** at lags 1–12 (last 2 hours), decaying
   gradually — confirming that recent traffic is a strong predictor of the next
   interval.
2. **Sharp daily peaks** at lags 144, 288, and 432 (1, 2, and 3 days), confirming
   that the same time-of-day in previous days is the most informative historical
   signal. This directly motivated the lag-144 and lag-288 features in XGBoost
   and the SEQ_LEN = 144 window for the LSTM.
3. **Persistent autocorrelation between daily peaks**, consistent with a slowly
   varying trend component.

### 4.4 Seasonal decomposition and stationarity (square 5161)

Figure 4 shows an additive seasonal decomposition with period = 144 (one day).

The **trend** rises gradually through November, peaks in mid-December, and dips
sharply around Christmas — relevant because the evaluation week (Dec 16–22) falls
just before this holiday dip. The **seasonal component** is stable and repeating,
with amplitude large relative to the residual, confirming that daily periodicity
accounts for the majority of predictable variation. The **residuals** are mostly
small but contain occasional large spikes corresponding to the anomalous bursts
in square 5161.

The Augmented Dickey-Fuller test returns ADF = −19.03 (p ≈ 0, critical value at
1%: −3.43), strongly rejecting the unit root null hypothesis. The series is
stationary in levels; no differencing is required. The same conclusion holds for
squares 4159 (ADF = −12.86) and 4556 (ADF = −14.20).

**Forecasting implications.** Strong stationarity, a dominant daily seasonal
component, and significant short-range autocorrelation make this series well-suited
to models that explicitly encode periodicity. The slowly varying trend and
anomalous residual spikes represent the main sources of forecasting difficulty,
particularly during the holiday period covered by the evaluation week.

---

## 5. Methodology

### 5.1 Forecasting setup

One-step-ahead forecasting: at each time step t, the model receives a history of
past observations and produces a single estimate of traffic at t+1. The evaluation
period is Monday 16 December to Sunday 22 December 2013 (1,008 steps × 10 min =
7 days). All models are trained exclusively on data prior to 16 December (6,480
steps, ~45 days). A validation split (Dec 9–15, 1,008 steps) was used for
hyperparameter selection; the test split (Dec 16–22) was held out until final
evaluation.

### 5.2 Model 1: SARIMA with Fourier regressors

**Architecture.** ARIMA(p, 0, q) errors with K pairs of sine/cosine exogenous
regressors encoding position within the daily cycle (period = 144 intervals).
d = 0 because the ADF test confirmed stationarity in levels. Native seasonal
ARIMA with s = 144 is computationally prohibitive in statsmodels (requires
estimating 144-lag seasonal polynomials); dynamic harmonic regression is the
standard alternative for high-frequency seasonal data (Hyndman & Athanasopoulos,
2021, §11.1).

**Input representation.** Full training series passed to the state-space fitter.
At inference, rolling one-step-ahead forecasts are produced by appending each
true observation to the Kalman filter state without refitting
(`append(..., refit=False)`).

**Preprocessing.** No scaling. Training via maximum-likelihood (L-BFGS-B).

**Hyperparameter tuning.** Grid search over ARIMA order (p, q) ∈
{(1,1),(1,2),(2,1),(2,2),(3,2),(2,3)} on the validation split:

| Order | Val MAE | Val RMSE | Notes |
|---|---|---|---|
| ARIMA(1,0,1) | 94.3 | 141.2 | Underfits short-range autocorrelation |
| ARIMA(1,0,2) | 89.7 | 135.8 | Marginal improvement |
| ARIMA(2,0,1) | 87.1 | 132.4 | Better AR term |
| **ARIMA(2,0,2)** | **83.6** | **127.9** | **Best — selected** |
| ARIMA(3,0,2) | 84.1 | 128.7 | No improvement over (2,0,2) |
| ARIMA(2,0,3) | 84.8 | 129.3 | Slight overfit |

ARIMA(2,0,2) selected: two AR lags capture the short-range autocorrelation
visible in the ACF, two MA terms handle residual correlation without overfitting.

### 5.3 Model 2: LSTM

**Architecture.** Single-layer LSTM with hidden size H, followed by a linear
output head mapping the final hidden state to a scalar prediction.

**Input representation.** Sliding window of SEQ_LEN consecutive scaled
observations fed as the input sequence. Window length chosen to match the
dominant ACF period.

**Preprocessing.** MinMaxScaler fitted on training split only; predictions
inverse-transformed before metric computation.

**Training.** Adam optimiser, MSE loss. Training dataset constructed by sliding
the window over the training series.

**Hyperparameter tuning — Round 1: sequence length**
(hidden=32, layers=1, epochs=15, lr=0.001)

| seq_len | Val MAE | Val RMSE | Reasoning |
|---|---|---|---|
| 72 | 118.4 | 172.3 | Half-day window misses daily lag |
| **144** | **109.2** | **161.8** | **Full day — matches ACF peak at lag 144** |
| 288 | 112.7 | 166.1 | Two days adds noise without benefit |

seq_len = 144 selected: aligns with the dominant daily periodicity.

**Round 2: hidden size** (seq_len=144, layers=1, epochs=15, lr=0.001)

| hidden | Val MAE | Val RMSE | Reasoning |
|---|---|---|---|
| 32 | 109.2 | 161.8 | Undercapacity for this series |
| **64** | **104.7** | **156.3** | **Best trade-off** |
| 128 | 105.9 | 157.8 | Marginal gain, higher cost |

hidden = 64 selected: sufficient capacity without overfitting on the small
training set.

**Round 3: epochs and learning rate** (seq_len=144, hidden=64, layers=1)

| epochs | lr | Val MAE | Val RMSE | Reasoning |
|---|---|---|---|---|
| 15 | 0.001 | 104.7 | 156.3 | Baseline |
| **20** | **0.001** | **102.1** | **153.4** | **More epochs improve convergence** |
| 20 | 0.0005 | 103.8 | 155.1 | Lower lr slows convergence, no benefit |

Final configuration: seq_len=144, hidden=64, epochs=20, lr=0.001.

### 5.4 Model 3: XGBoost

**Architecture.** Gradient-boosted regression trees treating forecasting as
supervised regression over a hand-crafted feature vector.

**Input representation.** For each time step t: recent lags (t−1 through
t−N_recent), daily lags (t−144, t−288, optionally t−1008), hour of day,
day of week, interval index within the day.

**Preprocessing.** No scaling (gradient boosting is scale-invariant). Single
batch training over all training-split feature rows.

**Hyperparameter tuning — Round 1: recent lag window**
(n_estimators=300, max_depth=5, lr=0.05)

| lags_recent | Val MAE | Val RMSE | Reasoning |
|---|---|---|---|
| 6 | 79.4 | 118.7 | Too short — misses 1-hour autocorrelation |
| **12** | **74.8** | **112.3** | **Best — 2 hours of recent context** |
| 24 | 75.6 | 113.1 | Marginal degradation from noise |

lags_recent = 12 selected.

**Round 2: n_estimators and max_depth** (lags_recent=12)

| n_est | depth | Val MAE | Val RMSE | Reasoning |
|---|---|---|---|---|
| 200 | 4 | 78.2 | 116.4 | Underfits |
| 300 | 5 | 74.8 | 112.3 | Good baseline |
| **500** | **6** | **72.1** | **109.8** | **Best** |
| 500 | 8 | 73.4 | 111.2 | Slight overfit at depth 8 |

n_estimators=500, max_depth=6 selected.

**Round 3: weekly lag** (n_estimators=500, max_depth=6)

| Daily lags | Val MAE | Val RMSE | Reasoning |
|---|---|---|---|
| [144, 288] | 72.1 | 109.8 | Baseline |
| [144, 288, 1008] | 71.3 | 108.6 | Marginal improvement from weekly lag |

Weekly lag (t−1008) retained: small but consistent improvement, and the
day-of-week statistics in Section 4 confirm a weekly pattern exists.

Final configuration: lags_recent=12, daily_lags=[144,288,1008],
n_estimators=500, max_depth=6, lr=0.05.

---

## 6. Results and Discussion

### 6.1 Quantitative results

**Square 5161 (highest traffic)**

| Model | MAE | MAPE (%) | RMSE | Train (s) | Inference (s) |
|---|---|---|---|---|---|
| SARIMA | 81.54 | 8.04 | 124.16 | 3.1 | 58.3 |
| LSTM | 108.17 | 12.06 | 158.71 | 54.0 | 1.0 |
| XGBoost | 88.12 | 8.42 | 132.81 | 1.5 | <0.1 |

**Square 5059**

| Model | MAE | MAPE (%) | RMSE | Train (s) | Inference (s) |
|---|---|---|---|---|---|
| SARIMA | 67.99 | 7.28 | 96.29 | 7.1 | 57.3 |
| LSTM | 91.21 | 8.37 | 129.01 | 61.4 | 1.0 |
| XGBoost | 71.40 | 6.97 | 103.33 | 1.5 | <0.1 |

**Square 5259**

| Model | MAE | MAPE (%) | RMSE | Train (s) | Inference (s) |
|---|---|---|---|---|---|
| SARIMA | 68.91 | 8.80 | 94.24 | 11.2 | 71.3 |
| LSTM | 76.10 | 8.00 | 110.87 | 90.7 | 1.0 |
| XGBoost | 65.71 | 6.99 | 96.86 | 1.6 | <0.1 |

**Timing averages across three squares**

| Model | Avg train (s) | Avg inference (s) |
|---|---|---|
| SARIMA | 7.1 | 62.3 |
| LSTM | 68.7 | 1.0 |
| XGBoost | 1.5 | <0.1 |

### 6.2 Comparative analysis

**SARIMA** achieves the lowest MAE and RMSE on two of three squares (5161 and
5059) and is competitive on 5259. This is consistent with the EDA findings:
the Fourier regressors directly encode the dominant daily seasonal structure
confirmed by the ACF, and the ARIMA(2,0,2) errors capture short-range
autocorrelation efficiently. The stationarity confirmed by ADF means no
differencing is needed, keeping the model parsimonious. SARIMA's main weakness
is its slow rolling inference (~62 s per square) due to 1,008 sequential Kalman
filter updates.

**XGBoost** is the best-performing model on square 5259 and consistently close
to SARIMA across all squares. The explicit lag-144 and lag-288 features directly
mirror the ACF structure, and the weekly lag (t−1008) provides a small additional
benefit consistent with the observed day-of-week pattern. XGBoost is by far the
fastest model: ~1.5 s to train and negligible inference time, making it the most
practical choice for frequent retraining scenarios.

**LSTM** underperforms both alternatives on all three squares. Several factors
explain this. First, the training set is small (~45 days, 6,480 steps); deep
models typically require substantially more data to outperform well-specified
statistical baselines (Makridakis et al., 2018). Second, the series is strongly
stationary and periodic — precisely the conditions under which SARIMA's explicit
seasonal parameterisation is most competitive. Third, despite tuning, the LSTM
was trained without a hyperparameter search over architecture depth or
regularisation; a deeper network with dropout or a longer training history could
improve results. The finding that LSTM underperforms here contrasts with Qiu et
al. (2023), who found LSTM superior on the same dataset — likely because their
study used a longer training window and applied clustering to group cells before
training, reducing within-group variance.

**Performance variation across squares.** All three models show consistent
relative rankings across the three squares, suggesting the differences are
structural rather than square-specific. Square 5161 produces the largest absolute
errors for all models, expected given its higher traffic magnitude and greater
variability (std = 1312.8 vs. ~950–1137 for the other two). MAPE values are
comparable across squares (6–12%), indicating proportional accuracy is similar.

### 6.3 Failure analysis

All three models struggle during periods of anomalous, event-driven traffic
spikes — sharp, short-duration bursts that deviate from the learned daily pattern.
These are visible in the prediction plots (figures/section4/) as intervals where
all three model traces remain near the expected seasonal level while the actual
series jumps abruptly. The maximum observed value in square 5161 during the
evaluation week is approximately 5× the mean, and all three models underpredict
these peaks substantially.

SARIMA cannot anticipate such spikes because its Fourier regressors encode only
the average seasonal shape. XGBoost relies on lag features that reflect normal
behaviour; a spike at t−1 will propagate into the next prediction, but the model
has no mechanism to anticipate the onset of a spike. The LSTM, trained on MSE
loss, learns to predict the conditional mean and is penalised for large
deviations, biasing it toward conservative predictions.

The holiday period (Dec 16–22) also presents a challenge: the seasonal
decomposition showed a trend dip beginning around Christmas, and all models
trained on November–early December data may slightly overpredict traffic during
this period as the trend shifts downward.

Possible mitigations include: anomaly detection as a pre-processing step to
flag and handle spike intervals separately; augmenting features with external
event calendars; or using quantile regression to produce prediction intervals
rather than point forecasts.

---

## 7. Conclusion and Future Work

This study compared SARIMA, LSTM, and XGBoost for one-step-ahead mobile network
traffic forecasting on the Telecom Italia Milan dataset. The main findings are:

1. **SARIMA with Fourier regressors** is the best-performing model on this
   dataset, achieving the lowest MAE and RMSE on two of three evaluation squares.
   Its advantage is directly attributable to the strong stationarity and clean
   daily periodicity confirmed by the ADF test and ACF analysis.

2. **XGBoost** is a strong practical alternative — nearly as accurate as SARIMA,
   far faster to train, and more easily extended with additional features. Its
   explicit daily-lag features (t−144, t−288, t−1008) directly encode the
   periodicity structure identified in the EDA.

3. **LSTM** underperforms both alternatives on this dataset, likely due to the
   small training set size (~45 days) and the strongly periodic, stationary
   nature of the series. Its advantage would likely emerge with a longer training
   history, multi-feature inputs, or a multi-step forecasting horizon.

4. **All three models fail on anomalous spikes**, which represent the hardest
   forecasting challenge and are not captured by any model that learns the average
   seasonal pattern.

**Limitations.** The training set covers only ~45 days, limiting the LSTM's
ability to generalise. Only one-step-ahead forecasting is evaluated; multi-step
horizons may favour different model architectures. No external features (weather,
events, SMS/call activity) are used.

**Future work.** Promising extensions include: (1) incorporating multi-step
forecasting to assess how errors accumulate; (2) adding SMS and call activity as
additional input features, which the dataset provides; (3) applying the hybrid
ARIMA–LSTM approach of Wang et al. (2022) to model linear and nonlinear
components separately; (4) using anomaly detection to pre-process spike intervals;
and (5) evaluating on a broader set of squares to assess generalisation across
the full spatial distribution.

---

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of
Milan and the Province of Trentino," *Scientific Data*, vol. 2, p. 150055, 2015.
https://doi.org/10.1038/sdata.2015.55

[2] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series
Analysis: Forecasting and Control*, 5th ed. Wiley, 2015.

[3] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in
*Proc. KDD 2016*, pp. 785–794. https://doi.org/10.1145/2939672.2939785

[4] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*,
3rd ed. OTexts, 2021. https://otexts.com/fpp3/

[5] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "Statistical and machine
learning forecasting methods: Concerns and ways forward," *PLOS ONE*, vol. 13,
no. 3, p. e0194889, 2018. https://doi.org/10.1371/journal.pone.0194889

[6] W. McKinney, *Python for Data Analysis*, 3rd ed. O'Reilly Media, 2022.
https://wesmckinney.com/book/

[7] C. Qiu et al., "LSTM vs GRU for mobile Internet traffic forecasting on the
Telecom Italia Milan dataset," *arXiv preprint*, 2023. [REPLACE WITH EXACT DOI]

[8] X. Shi et al., "A survey of deep learning for network traffic forecasting,"
*ACM Computing Surveys*, vol. 54, no. 2, 2021.
https://doi.org/10.1145/3447556

[9] H. D. Trinh et al., "Enabling mobile traffic forecasting with deep learning,"
in *Proc. ACM MSWiM 2018*. [REPLACE WITH EXACT DOI]

[10] H. Wang et al., "Hybrid double-seasonal ARIMA–LSTM model for cellular
traffic forecasting," *Scientific Reports*, vol. 12, p. 8342, 2022.
[REPLACE WITH EXACT DOI]

GitHub repository: https://github.com/Munana122/milan-traffic-forecasting
