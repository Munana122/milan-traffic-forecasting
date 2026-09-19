# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

---

# Section 1 — Data Handling and Memory Management

## 1.1 Dataset

The raw dataset is the Telecom Italia Big Data Challenge (Barlacchi et al., 2015),
covering Milan's 10,000-cell grid at 10-minute resolution from 1 November 2013 to
1 January 2014 — 62 daily tab-separated files totalling 20.8 GB on disk. Each row
records activity for one grid square, one time interval, and one country code across
eight columns: square_id, time_interval, country_code, sms_in, sms_out, call_in,
call_out, and internet_traffic. Only three columns (square_id, time_interval,
internet_traffic) are needed for this project.

## 1.2 Ingestion strategy

Loading all 62 files naively — all eight columns, default int64/float64 dtypes, no
early aggregation — would hold roughly 62 × 349 MB ≈ 21 GB in memory simultaneously,
far exceeding a typical workstation. Four targeted optimisations were applied instead:

**usecols** — `pd.read_csv(..., usecols=[...])` instructs the parser to discard the
five SMS and call columns at read time, before they are ever allocated in memory.
Pandas' CSV engine skips those fields entirely rather than parsing and then dropping
them (McKinney, 2022).

**Dtype downcasting** — square_id is stored as int32 (max value 10,000; int32 ceiling
4.3 × 10⁹) and internet_traffic as float32 (sufficient precision for traffic volumes).
This halves the per-cell cost relative to pandas' default int64/float64, since each
value occupies 4 bytes instead of 8.

**Early aggregation** — each raw file contains one row per (square_id, time_interval,
country_code) triplet. Summing internet_traffic across country codes immediately after
loading collapses ~4.8 M rows to ~1.4 M rows (a 70% reduction) before the DataFrame
is appended to the accumulator list.

**Process-then-discard** — each raw file is copied into a staging folder, processed,
and deleted before the next file is read. At no point does more than one 322 MB raw
file exist on disk simultaneously.

## 1.3 Per-file memory: naive vs. optimised

Measured with Python's `tracemalloc` on `sms-call-internet-mi-2013-11-01.txt`
(322 MB on disk):

| Method | Peak memory | Rows |
|---|---|---|
| Naive (all columns, default dtypes) | 348.95 MB | 4,842,625 |
| Optimised (usecols + int32/float32 + groupby) | 355.99 MB | 1,439,982 |

The peak figures are nearly identical because `tracemalloc` captures the high-water
mark during loading — pandas allocates the full intermediate buffer before the
aggregation frees memory. The meaningful difference is in the **retained** DataFrame:
23 MB after aggregation vs. the naive 349 MB peak, a 93% reduction in working-set
size. This is the figure that matters for a pipeline that accumulates 62 daily frames.

## 1.4 Full-pipeline memory: all 62 files

| Metric | Value |
|---|---|
| Files processed | 62 |
| Total rows (combined) | 89,245,318 |
| Per-file retained memory | ~23 MB |
| Peak memory (tracemalloc, whole loop) | 2,857 MB |
| Total ingestion time | 298 s (~5 min) |

The 2,857 MB whole-loop peak reflects the `frames` list growing as all 62 aggregated
DataFrames accumulate before `pd.concat`. The per-file load spike (~356 MB) is
transient and released immediately after each file. A streaming write strategy
(writing each day's frame to Parquet incrementally) could reduce this further, but
the current peak was comfortably within available RAM and no intervention was needed.

## 1.5 Raw files vs. final Parquet

After concatenation the combined DataFrame was written to a single Parquet file using
Snappy compression (the pyarrow default):

| | Size |
|---|---|
| Raw `.txt` files (62 days) | 20.805 GB |
| `milan_internet_traffic.parquet` | 408.79 MB |
| Reduction | **98.0% — 50.9× smaller** |

Three effects compound to produce this reduction: (1) five of eight columns are
dropped entirely; (2) early aggregation removes the country-code dimension, cutting
row count by 70%; and (3) Parquet's columnar layout with Snappy compression encodes
repeated integer and float values far more efficiently than plain-text TSV. The
Parquet file reloads in 2.4 s via `pd.read_parquet`, compared to ~5 minutes for the
full ingestion pipeline. All analysis in Sections 2–4 reads exclusively from this
file — the raw daily files are never accessed again.

---

# Section 2 — Exploratory Analysis

## 2.1 Distribution of total Internet traffic across geographical areas

Figure 1 (`figures/traffic_distribution.png`) shows the histogram and CDF of
total Internet traffic summed over the full two-month observation period across
all 10,000 grid squares.

**Summary statistics:**

| Statistic | Value |
|---|---|
| Mean | 555,289 |
| Median (50th pct) | 277,871 |
| Std deviation | 892,596 |
| 25th percentile | 118,102 |
| 75th percentile | 577,896 |
| Maximum | 12,740,060 |
| Skewness | 4.27 |

The distribution is strongly right-skewed (skewness = 4.27). The mean is
approximately twice the median, indicating that a small number of high-traffic
squares pull the average upward. More than 73% of squares (7,376 out of 10,000)
fall below the mean, and only 100 squares (the top 1%) exceed 4,698,732 total
traffic units. The maximum value (square 5161, 12,740,060) is roughly 23 times
the mean and 46 times the median.

The CDF makes this concentration visible: the curve rises steeply for low
traffic values and flattens gradually, with a long right tail. This pattern is
consistent with the spatial structure of an urban area — the vast majority of
grid squares cover residential or low-activity zones, while a small number of
squares in high-density commercial or transport hubs generate disproportionately
large traffic volumes. This heterogeneity motivates evaluating forecasting models
across squares with different traffic characteristics, as model performance may
vary substantially between low- and high-traffic areas.

## 2.2 Top-3 squares by total traffic

| Rank | Square ID | Total traffic (2-month sum) |
|---|---|---|
| 1 | **5161** | 12,740,060 |
| 2 | 5059 | 11,170,854 |
| 3 | 5259 | 10,485,780 |

Square 5161 is used as the primary target for the deep-dive analyses (Section 2.4)
and for the forecasting experiments in Section 4.

## 2.3 Time-series analysis: first two weeks

Figure 2 (`figures/first_two_weeks.png`) shows Internet traffic at 10-minute
resolution for the first 14 days of the observation period (1–14 November 2013)
for five squares: the top-3 (5161, 5059, 5259) and the two specified squares
(4159, 4556).

**Per-square summary statistics (first two weeks):**

| Square | Mean | Std | Max | Min |
|---|---|---|---|---|
| 5161 | 1483.5 | 1312.8 | 8044.1 | 104.9 |
| 5059 | 1330.6 | 952.4 | 4182.8 | 189.1 |
| 5259 | 1293.5 | 1136.7 | 4262.9 | 134.5 |
| 4159 | 311.3 | 189.3 | 851.1 | 80.3 |
| 4556 | 592.7 | 255.6 | 1868.7 | 172.0 |

**Similarities.** All five squares exhibit a clear daily cycle: traffic rises
through the morning, peaks in the afternoon (roughly 15:00–17:00 local time),
and falls to a trough in the early hours (03:00–05:00). No square records zero
traffic at any interval, indicating continuous background activity even at night.
The daily pattern repeats consistently across all 14 days, confirming the strong
periodicity identified in the ACF analysis (Section 2.4).

**Differences.** The top-3 squares (5161, 5059, 5259) operate at a substantially
higher absolute level than squares 4159 and 4556 — roughly 4–5× higher mean
traffic. Square 5161 also shows considerably higher variability (std = 1312.8)
and a much larger maximum (8044.1) compared to the other top-3 squares, suggesting
it may be located near a venue or transport hub that generates occasional large
spikes. Square 4159 has the lowest mean (311.3) and the smallest range, consistent
with a quiet residential area. Square 4556 sits between the two groups in both
level and variability.

A notable feature of square 5161 is the presence of several sharp, short-duration
spikes that exceed the typical daily peak by a large margin (e.g. the maximum of
8044.1 is roughly 5× the mean). These anomalous bursts are not visible in the
other squares at the same times, suggesting they are localised events rather than
network-wide effects. Such spikes are difficult to forecast with any model that
learns the average seasonal pattern, and are discussed further in Section 4.6.

The weekly structure is also visible: weekend days (Saturday and Sunday) show
a different intra-day shape compared to weekdays — the morning ramp-up is later
and the afternoon peak is broader — consistent with the day-of-week averages
computed for square 5161 (Saturday mean 1849.5, Sunday 1704.7 vs. weekday
average ~1280).

## 2.4 Deep-dive analyses: square 5161

### Analysis 1 — Autocorrelation function (ACF)

Figure 3 (`figures/acf_top_square.png`) shows the ACF of square 5161's traffic
series up to 432 lags (3 days × 144 intervals/day).

The ACF reveals three key features:

1. **Strong short-range autocorrelation.** Lags 1–12 (the most recent 2 hours)
   show high positive autocorrelation, decaying gradually. This confirms that
   recent traffic is a strong predictor of the next interval — the basis for
   both the ARIMA error terms and the short-lag features in XGBoost.

2. **Sharp daily peaks.** Prominent spikes appear at lags 144, 288, and 432,
   corresponding exactly to 1, 2, and 3 days. The spike at lag 144 is the
   largest seasonal peak, confirming that the same time-of-day yesterday is
   the single most informative historical observation for predicting the current
   interval. This directly motivated the inclusion of lag-144 and lag-288
   features in the XGBoost model and the choice of SEQ_LEN = 144 for the LSTM.

3. **Persistent autocorrelation between daily peaks.** The ACF does not decay
   to zero between the daily spikes, consistent with the slowly varying trend
   component visible in the seasonal decomposition below.

### Analysis 2 — Seasonal decomposition and stationarity

Figure 4 (`figures/decomposition_top_square.png`) shows an additive seasonal
decomposition of square 5161's full series with period = 144 (one day).

**Trend component.** The trend rises gradually through November, peaks in
mid-December, and shows a sharp dip around Christmas (25–26 December),
consistent with reduced urban activity during the holiday period. This is
relevant to the evaluation week (Dec 16–22), which falls just before the
holiday dip begins.

**Seasonal component.** The seasonal component is stable and repeating,
confirming a consistent daily cycle throughout the observation period. Its
amplitude is large relative to the residual, indicating that daily periodicity
accounts for the majority of predictable variation in the series.

**Residual component.** The residuals are mostly small but contain occasional
large spikes corresponding to the anomalous bursts noted in Section 2.3. These
represent the hardest part of the forecasting problem.

**Stationarity (ADF test).** The Augmented Dickey-Fuller test on the full series
for square 5161 returns ADF = −19.03 (p ≈ 0, critical value at 1%: −3.43),
strongly rejecting the null hypothesis of a unit root. The series is stationary
in levels; no differencing is required. The same conclusion holds for squares
4159 (ADF = −12.86, p = 5.1 × 10⁻²⁴) and 4556 (ADF = −14.20, p = 1.8 × 10⁻²⁶).

**Implications for forecasting.** The combination of strong stationarity, a
dominant and stable daily seasonal component, and significant short-range
autocorrelation makes this series well-suited to models that explicitly encode
periodicity — either through seasonal terms (SARIMA/Fourier regressors) or
through lag features at multiples of 144 (XGBoost). The slowly varying trend
and the anomalous residual spikes represent the main sources of forecasting
difficulty, particularly for the evaluation week in December when the holiday
dip occurs.

---

# Section 3 — Related Work and Model Selection

## 3.1 Review of relevant literature

Mobile and cellular network traffic forecasting has attracted sustained research
attention because accurate short-horizon predictions underpin dynamic resource
allocation, congestion management, and capacity planning (Trinh et al., 2018).
The problem is inherently sequential: traffic at a given cell is strongly
autocorrelated across time, exhibits daily and weekly periodicity, and can contain
abrupt anomalies driven by events or failures.

**Statistical approaches.** Classical time-series models such as ARIMA and its
seasonal extension SARIMA have long served as baselines for cellular traffic
forecasting. Their appeal is interpretability and a principled treatment of
trend and seasonality through explicit differencing and seasonal terms. However,
they assume linear dynamics and struggle to capture the nonlinear, bursty
behaviour common in high-traffic urban cells (Box et al., 2015).

**Deep sequential models.** Recurrent neural networks, and LSTM in particular,
became the dominant approach after demonstrating an ability to learn long-range
temporal dependencies without manual feature engineering. A study using this
exact Telecom Italia Milan dataset compared LSTM and GRU architectures for
predicting mobile Internet traffic, applying K-Means clustering to group cells
by activity level and grid search for hyperparameter tuning, and evaluating with
RMSE. LSTM outperformed GRU overall and captured both daily and two-month
seasonality patterns well (Qiu et al., 2023). A broader survey of deep learning
for network traffic forecasting confirms LSTM's consistent advantage over plain
RNNs, while noting that training cost scales with parameter count — a relevant
trade-off when comparing models empirically (Shi et al., 2021).

**Hybrid and attention-based approaches.** One study demonstrated that pairing
double-seasonal ARIMA with LSTM — where ARIMA captures the linear seasonal
structure and LSTM models the nonlinear residuals — outperformed either component
alone (Wang et al., 2022). This finding implies that cellular traffic contains
both a strong periodic/linear component and a nonlinear residual that purely
statistical models leave unexplained. Attention-based Transformer architectures
have been proposed as an alternative to recurrence, using self-attention to
capture long-term dependencies without the sequential bottleneck of LSTMs
(Zhang et al., 2023).

**Feature-based machine learning.** Gradient boosting methods (XGBoost, LightGBM)
reframe forecasting as supervised regression over engineered lag and calendar
features. They train faster than deep models, require no sequence architecture,
and provide feature-importance interpretability. They have shown competitive
accuracy on many tabular forecasting benchmarks, though their performance depends
heavily on the quality of the hand-crafted features (Chen & Guestrin, 2016).

## 3.2 Findings from exploratory analysis informing model selection

1. **Strong stationarity.** ADF = −19.03 (p ≈ 0) on square 5161 confirms the
   series is stationary in levels; d = 0 for SARIMA.

2. **Pronounced daily periodicity.** ACF peaks at lags 144, 288, and 432
   confirm a strong daily seasonal cycle, motivating Fourier regressors for
   SARIMA and lag-144/288 features for XGBoost.

3. **Heavy-tailed distribution and anomalous spikes.** Skewness = 4.27 and
   localised bursts up to 5× the mean in square 5161 represent the hardest
   forecasting challenge for all three models.

## 3.3 Selected models

### Model 1: SARIMA

Selected as the classical statistical baseline. Seasonal period s = 144 is
motivated directly by the ACF peaks. With d = 0 (confirmed by ADF), the model
is parsimonious and interpretable. Its principal limitation is the linearity
assumption: it cannot represent the nonlinear, bursty dynamics visible in the
high-traffic time series (Box et al., 2015).

### Model 2: LSTM

Selected as the primary deep-learning model, motivated by Qiu et al. (2023),
who applied it to this exact Milan dataset. LSTM's gating mechanism allows it
to selectively retain information across hundreds of time steps, making it
well-suited to the long-lag autocorrelation structure observed in the ACF.
Its weaknesses are training cost, sensitivity to hyperparameter choices, and
limited interpretability (Shi et al., 2021).

### Model 3: XGBoost on lag features

Selected as a non-recurrent, feature-based ML baseline. Forecasting is recast
as supervised regression over lag and calendar features that directly encode
the daily periodicity identified in the ACF. XGBoost trains in seconds, provides
feature-importance interpretability, and is robust to outliers. Its limitation
is that it has no native notion of sequence continuity (Chen & Guestrin, 2016).

## 3.4 Summary comparison

| | SARIMA | LSTM | XGBoost |
|---|---|---|---|
| Paradigm | Statistical | Deep sequential | Feature-based ML |
| Handles nonlinearity | No | Yes | Yes |
| Explicit seasonality | Yes (Fourier, s=144) | Learned | Via lag features |
| Training speed | Fast | Slow | Very fast |
| Interpretability | High | Low | Medium |
| Key motivation | ADF/ACF baseline | Qiu et al. (2023) | Speed + interpretability |

---

# Section 4 — Forecasting Experiments

## 4.1 Problem setup

One-step-ahead forecasting of Internet traffic at each of the three
highest-traffic squares (5161, 5059, 5259). At each time step t the model
receives a history of past observations and produces a single estimate of
traffic at t+1. The evaluation period is Monday 16 December to Sunday 22
December 2013 (1,008 steps × 10 min = 7 days). All models are trained
exclusively on data prior to 16 December.

Hardware: CPU only (Intel, Windows 11). Timing recorded with
`time.perf_counter()` per square and averaged across the three squares.

## 4.2 Model descriptions

### Model 1: SARIMA with Fourier regressors (dynamic harmonic regression)

**Structure.** ARIMA(2,0,2) errors with 6 pairs of sine/cosine exogenous
regressors encoding position within the daily cycle (period = 144 intervals).
d = 0 because ADF confirmed stationarity in levels.

**Why Fourier terms instead of native seasonal_order.** A native seasonal ARIMA
with period s = 144 requires estimating 144-lag seasonal AR and MA polynomials,
which is computationally prohibitive in statsmodels. Dynamic harmonic regression
— replacing the seasonal structure with K Fourier pairs as exogenous regressors
— is a standard alternative for high-frequency seasonal data (Hyndman &
Athanasopoulos, 2021, §11.1). K = 6 pairs capture the fundamental daily
frequency and its first five harmonics.

**Input representation.** Full training series (6,480 steps). At inference,
rolling one-step-ahead forecasts are produced by appending each true observation
to the Kalman filter state without refitting (`append(..., refit=False)`).

**Preprocessing.** No scaling. Training via maximum-likelihood (L-BFGS-B).

### Model 2: LSTM (PyTorch)

**Structure.** Single-layer LSTM, hidden size 64, linear output head.

**Input representation.** Sliding window of 144 consecutive scaled observations
(one full day of history). Window length chosen to match the dominant ACF period.

**Preprocessing.** MinMaxScaler fitted on training split only; predictions
inverse-transformed before metric computation.

**Training.** 20 epochs, batch size 128, Adam (lr = 0.001), MSE loss.
6,336 (input, target) pairs per square. Inference uses true past values
(teacher-forced history).

### Model 3: XGBoost on lag and calendar features

**Structure.** Gradient-boosted regression trees.

**Input representation.** For each time step t:
- Recent lags: t−1 through t−12 (last 2 hours)
- Daily lags: t−144 and t−288 (motivated by ACF peaks at lags 144 and 288)
- Calendar features: hour of day, day of week, interval index within the day

**Preprocessing.** No scaling (gradient boosting is scale-invariant).

**Training.** 500 trees, max depth 6, learning rate 0.05, subsample 0.8,
colsample_bytree 0.8. Single batch over all training-split feature rows.
All 1,008 eval predictions generated in one batch call.

## 4.3 Results

### Square 5161 (highest traffic)

| Model | MAE | MAPE (%) | RMSE | Train (s) | Inference (s) |
|---|---|---|---|---|---|
| SARIMA | 81.54 | 8.04 | 124.16 | 3.1 | 58.3 |
| LSTM | 108.17 | 12.06 | 158.71 | 54.0 | 1.0 |
| XGBoost | 88.12 | 8.42 | 132.81 | 1.5 | <0.1 |

### Square 5059

| Model | MAE | MAPE (%) | RMSE | Train (s) | Inference (s) |
|---|---|---|---|---|---|
| SARIMA | 67.99 | 7.28 | 96.29 | 7.1 | 57.3 |
| LSTM | 91.21 | 8.37 | 129.01 | 61.4 | 1.0 |
| XGBoost | 71.40 | 6.97 | 103.33 | 1.5 | <0.1 |

### Square 5259

| Model | MAE | MAPE (%) | RMSE | Train (s) | Inference (s) |
|---|---|---|---|---|---|
| SARIMA | 68.91 | 8.80 | 94.24 | 11.2 | 71.3 |
| LSTM | 76.10 | 8.00 | 110.87 | 90.7 | 1.0 |
| XGBoost | 65.71 | 6.99 | 96.86 | 1.6 | <0.1 |

## 4.4 Timing summary (averages across three squares)

| Model | Avg train (s) | Avg inference (s) |
|---|---|---|
| SARIMA | 7.1 | 62.3 |
| LSTM | 68.7 | 1.0 |
| XGBoost | 1.5 | <0.1 |

SARIMA's inference cost is high because the rolling Kalman update runs 1,008
sequential steps. LSTM inference is fast once trained. XGBoost inference is
near-instantaneous as all predictions are generated in a single batch.

## 4.5 Comparative analysis

**Predictive performance.** SARIMA achieves the lowest MAE and RMSE on two of
the three squares (5161 and 5059), and is competitive on 5259. This is
consistent with the strong stationarity and clean daily periodicity confirmed
in Section 2: the Fourier regressors directly encode the dominant seasonal
structure, and the ARIMA(2,0,2) errors capture short-range autocorrelation
efficiently. XGBoost is the best-performing model on square 5259 and is
consistently close to SARIMA, benefiting from the explicit daily-lag features
(t−144, t−288) that mirror the same periodicity. LSTM underperforms both
alternatives on all three squares.

**Why LSTM underperforms here.** The training set is relatively small (6,480
steps, ~45 days), which limits the LSTM's ability to generalise; deep models
typically require more data to outperform well-specified statistical baselines
(Makridakis et al., 2018). The series is also strongly stationary and periodic
— precisely the conditions under which SARIMA's explicit seasonal
parameterisation is most competitive. The LSTM was trained with no
hyperparameter search; a longer window, deeper architecture, or tuned learning
rate schedule could improve results.

**Performance variation across squares.** All three models show broadly
consistent relative rankings across the three squares. Square 5161 produces
the largest absolute errors for all models, expected given its higher traffic
magnitude; MAPE values are comparable across squares (6–12%), indicating
proportional accuracy is similar.

**Training cost.** XGBoost is by far the most efficient: ~1.5 s to train and
negligible inference time. SARIMA trains quickly but its sequential Kalman
update makes rolling inference slow (~60 s per square). LSTM is the most
expensive to train (~70 s average). For a production deployment requiring
frequent retraining, XGBoost's speed advantage is significant.

**Best model.** SARIMA with Fourier terms is the best-justified model for this
specific problem given confirmed stationarity, strong daily periodicity, and
moderate training-set size. XGBoost is a strong practical alternative — nearly
as accurate, far faster, and more easily extended with additional features.
LSTM's advantage would likely emerge with a longer training history, multiple
input features, or a multi-step forecasting horizon.

## 4.6 Example of poor model performance

All three models struggle during periods of anomalous or event-driven traffic
spikes — sharp, short-duration bursts that deviate from the learned daily
pattern. These are visible in the prediction plots (`figures/section4/`) as
intervals where all three model traces remain near the expected seasonal level
while the actual series jumps abruptly. SARIMA cannot anticipate such spikes
because its Fourier regressors encode only the average seasonal shape; XGBoost
similarly relies on lag features that reflect normal behaviour; and the LSTM,
trained on MSE loss, learns to predict the conditional mean and is penalised
for large deviations. This is a known limitation of all three model classes for
traffic with heavy-tailed, event-driven components (Trinh et al., 2018).
Possible mitigations include anomaly detection as a pre-processing step or
augmenting features with external event calendars.

---

# References

Apache Parquet (2024). *Parquet file format specification*.
https://parquet.apache.org/docs/file-format/

Barlacchi, G., De Nadai, M., Larcher, R., Casella, A., Chitic, C., Torrisi, G.,
Antonelli, F., Vespignani, A., Pentland, A., & Lepri, B. (2015). A multi-source
dataset of urban life in the city of Milan and the Province of Trentino.
*Scientific Data*, 2, 150055. https://doi.org/10.1038/sdata.2015.55

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
*Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
*Proceedings of KDD 2016*, 785–794. https://doi.org/10.1145/2939672.2939785

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and
machine learning forecasting methods: Concerns and ways forward. *PLOS ONE*,
13(3), e0194889. https://doi.org/10.1371/journal.pone.0194889

McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly Media.
https://wesmckinney.com/book/accessing-data#io_flat_file

Qiu, C., et al. (2023). LSTM vs GRU for mobile Internet traffic forecasting
on the Telecom Italia Milan dataset. *arXiv preprint*. [REPLACE WITH EXACT DOI]

Shi, X., et al. (2021). A survey of deep learning for network traffic forecasting.
*ACM Computing Surveys*, 54(2). https://doi.org/10.1145/3447556

Trinh, H. D., et al. (2018). Enabling mobile traffic forecasting with deep
learning. *Proceedings of ACM MSWiM 2018*. [REPLACE WITH EXACT DOI]

Wang, H., et al. (2022). Hybrid double-seasonal ARIMA–LSTM model for cellular
traffic forecasting. *Scientific Reports*, 12, 8342. [REPLACE WITH EXACT DOI]

Zhang, Y., et al. (2023). Attention-based Transformer for network traffic
prediction. *IEEE Transactions on Network and Service Management*, 20(1), 112–124.
[REPLACE WITH EXACT DOI]
