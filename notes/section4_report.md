# Section 4 — Forecasting Experiments

## 4.1 Problem setup

One-step-ahead forecasting of Internet traffic at each of the three
highest-traffic squares (5161, 5059, 5259) identified in Section 2.
At each time step t the model receives a history of past observations and
produces a single estimate of traffic at t+1. The evaluation period is
Monday 16 December to Sunday 22 December 2013 (1,008 steps × 10 min = 7 days).
All models are trained exclusively on data prior to 16 December; no future
information leaks into training.

Hardware: all experiments run on CPU (Intel, Windows 11).
Timing statistics are reported per square and averaged across the three squares.

---

## 4.2 Model descriptions

### Model 1: SARIMA with Fourier regressors (dynamic harmonic regression)

**Structure.** ARIMA(2,0,2) errors with 6 pairs of sine/cosine exogenous
regressors encoding position within the daily cycle (period = 144 intervals).
The non-seasonal differencing order d is set to 0 because the ADF test on
square 5161 returned a statistic of −19.03 (p ≈ 0), confirming stationarity
in levels.

**Why Fourier terms instead of native seasonal_order.** A native seasonal ARIMA
with period s = 144 requires estimating 144-lag seasonal AR and MA polynomials,
which is computationally prohibitive in statsmodels. Dynamic harmonic regression
— replacing the seasonal structure with K Fourier pairs as exogenous regressors
— is a standard, well-cited alternative for high-frequency seasonal data
(Hyndman & Athanasopoulos, 2021, §11.1). K = 6 pairs were used, capturing the
fundamental daily frequency and its first five harmonics.

**Input representation.** The full training series (6,480 steps, ~45 days) is
passed to the state-space fitter. At inference, a rolling one-step-ahead
forecast is produced by appending each true observation to the Kalman filter
state without refitting (statsmodels `append(..., refit=False)`), so the model
updates its state estimate at each step using the true value.

**Preprocessing.** No scaling; SARIMA operates on raw traffic values.

**Training procedure.** Maximum-likelihood estimation via the Kalman filter
(L-BFGS-B optimiser, statsmodels default).

---

### Model 2: LSTM (PyTorch)

**Structure.** Single-layer LSTM with hidden size 64, followed by a linear
output head mapping the final hidden state to a scalar prediction.

**Input representation.** A sliding window of SEQ_LEN = 144 consecutive
scaled observations (one full day of history) is fed as the input sequence
at each step. The window length was chosen to match the dominant daily
periodicity identified in the ACF (Section 2).

**Preprocessing.** MinMaxScaler fitted on the training split only, applied to
both train and eval. Predictions are inverse-transformed before metric
computation.

**Training procedure.** 20 epochs, batch size 128, Adam optimiser (lr = 0.001),
MSE loss. The training dataset is constructed by sliding the 144-step window
over the training series, yielding 6,336 (input, target) pairs per square.

**Inference.** At each eval step, the window is taken from the concatenated
train+eval series using true past values (teacher-forced history), producing
a true one-step-ahead forecast.

---

### Model 3: XGBoost on lag and calendar features

**Structure.** Gradient-boosted regression trees (XGBoost), treating
forecasting as supervised regression over a hand-crafted feature vector.

**Input representation.** For each time step t, the feature vector contains:
- Recent lags: traffic at t−1 through t−12 (last 2 hours)
- Daily lags: traffic at t−144 and t−288 (yesterday and two days ago at the
  same time of day) — directly motivated by the ACF peaks at lags 144 and 288
- Calendar features: hour of day, day of week, interval index within the day

**Preprocessing.** No scaling; gradient boosting is scale-invariant.

**Training procedure.** 500 trees, max depth 6, learning rate 0.05, subsample
0.8, colsample_bytree 0.8. Trained in a single batch on all training-split
feature rows.

**Inference.** All 1,008 eval-period predictions are generated in a single
batch call (no sequential state update), using true lag values from the
concatenated series.

---

## 4.3 Results

### Square 5161 (highest traffic)

| Model   | MAE     | MAPE (%) | RMSE    | Train (s) | Inference (s) |
|---------|---------|----------|---------|-----------|---------------|
| SARIMA  | 81.54   | 8.04     | 124.16  | 3.1       | 58.3          |
| LSTM    | 108.17  | 12.06    | 158.71  | 54.0      | 1.0           |
| XGBoost | 88.12   | 8.42     | 132.81  | 1.5       | <0.1          |

### Square 5059

| Model   | MAE    | MAPE (%) | RMSE   | Train (s) | Inference (s) |
|---------|--------|----------|--------|-----------|---------------|
| SARIMA  | 67.99  | 7.28     | 96.29  | 7.1       | 57.3          |
| LSTM    | 91.21  | 8.37     | 129.01 | 61.4      | 1.0           |
| XGBoost | 71.40  | 6.97     | 103.33 | 1.5       | <0.1          |

### Square 5259

| Model   | MAE    | MAPE (%) | RMSE   | Train (s) | Inference (s) |
|---------|--------|----------|--------|-----------|---------------|
| SARIMA  | 68.91  | 8.80     | 94.24  | 11.2      | 71.3          |
| LSTM    | 76.10  | 8.00     | 110.87 | 90.7      | 1.0           |
| XGBoost | 65.71  | 6.99     | 96.86  | 1.6       | <0.1          |

---

## 4.4 Timing summary

Training and inference times were recorded with `time.perf_counter()` on the
same hardware for all three squares. The figures below are averages across
the three squares.

| Model   | Avg train (s) | Avg inference (s) |
|---------|---------------|-------------------|
| SARIMA  | 7.1           | 62.3              |
| LSTM    | 68.7          | 1.0               |
| XGBoost | 1.5           | <0.1              |

SARIMA's inference cost is high because the rolling one-step-ahead loop
updates the Kalman filter state 1,008 times sequentially. LSTM inference
is fast once trained because all 1,008 predictions are independent forward
passes. XGBoost inference is near-instantaneous as all predictions are
generated in a single batch.

---

## 4.5 Comparative analysis

**Predictive performance.** SARIMA achieves the lowest MAE and RMSE on two of
the three squares (5161 and 5059), and is competitive on 5259. This is
consistent with the strong stationarity and clean daily periodicity confirmed
in Section 2: the Fourier regressors directly encode the dominant seasonal
structure, and the ARIMA(2,0,2) errors capture short-range autocorrelation
efficiently. XGBoost is the best-performing model on square 5259 and is
consistently close to SARIMA, benefiting from the explicit daily-lag features
(t−144, t−288) that mirror the same periodicity. LSTM underperforms both
alternatives on all three squares in this experiment, despite being the model
most commonly reported as superior in the literature (Qiu et al., 2023).

**Why LSTM underperforms here.** Several factors likely contribute. First, the
training set is relatively small (6,480 steps, ~45 days), which limits the
LSTM's ability to generalise; deep models typically require more data to
outperform well-specified statistical baselines (Makridakis et al., 2018).
Second, the series is strongly stationary and periodic — precisely the
conditions under which SARIMA's explicit seasonal parameterisation is most
competitive. Third, the LSTM was trained with a fixed 144-step window and
no hyperparameter search; a longer window, deeper architecture, or tuned
learning rate schedule could improve results.

**Performance variation across squares.** All three models show broadly
consistent relative rankings across the three squares, suggesting the
differences are structural rather than square-specific. Square 5161 (the
highest-traffic cell) produces the largest absolute errors for all models,
which is expected given its higher traffic magnitude; MAPE values are
comparable across squares (6–12%), indicating proportional accuracy is
similar.

**Training cost.** XGBoost is by far the most efficient: ~1.5 s to train and
negligible inference time. SARIMA trains quickly but its sequential Kalman
update makes rolling inference slow (~60 s per square). LSTM is the most
expensive to train (~70 s average) and requires careful scaling and
hyperparameter choices. For a production deployment requiring frequent
retraining, XGBoost's speed advantage is significant.

**Suitability for this dataset.** Given the confirmed stationarity, strong
daily periodicity, and moderate training-set size, SARIMA with Fourier terms
is the best-justified model for this specific problem. XGBoost is a strong
practical alternative — nearly as accurate, far faster, and more easily
extended with additional features. LSTM's advantage would likely emerge with
a longer training history, multiple input features (e.g. SMS and call
activity), or a multi-step forecasting horizon where recurrent memory becomes
more valuable.

---

## 4.6 Example of poor model performance

All three models struggle during periods of anomalous or event-driven traffic
spikes — sharp, short-duration bursts that deviate from the learned daily
pattern. These are visible in the prediction plots as intervals where all
three model traces remain near the expected seasonal level while the actual
series jumps abruptly. SARIMA cannot anticipate such spikes because its
Fourier regressors encode only the average seasonal shape; XGBoost similarly
relies on lag features that reflect normal behaviour; and the LSTM, trained
on MSE loss, learns to predict the conditional mean and is penalised for
large deviations. This is a known limitation of all three model classes for
traffic with heavy-tailed, event-driven components (Trinh et al., 2018).
Possible mitigations include anomaly detection as a pre-processing step,
or augmenting features with external event calendars.

---

## References

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and
machine learning forecasting methods: Concerns and ways forward. *PLOS ONE*,
13(3), e0194889. https://doi.org/10.1371/journal.pone.0194889

Qiu, C., et al. (2023). LSTM vs GRU for mobile Internet traffic forecasting
on the Telecom Italia Milan dataset. *arXiv preprint*.

Trinh, H. D., et al. (2018). Enabling mobile traffic forecasting with deep
learning. *Proceedings of ACM MSWiM 2018*.
