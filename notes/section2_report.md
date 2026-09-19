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

---

## 2.2 Top-3 squares by total traffic

| Rank | Square ID | Total traffic (2-month sum) |
|---|---|---|
| 1 | **5161** | 12,740,060 |
| 2 | 5059 | 11,170,854 |
| 3 | 5259 | 10,485,780 |

Square 5161 is used as the primary target for the deep-dive analyses (Section
2.3) and for the forecasting experiments in Section 4.

---

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

---

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

3. **No evidence of a weekly seasonal component** at this lag range (3 days is
   insufficient to observe a full weekly cycle at lag 1008), though the
   day-of-week statistics in Section 2.3 suggest a weekly pattern exists.

The ACF does not decay to zero, which is consistent with the presence of a
slowly varying trend component visible in the seasonal decomposition below.

### Analysis 2 — Seasonal decomposition and stationarity

Figure 4 (`figures/decomposition_top_square.png`) shows an additive seasonal
decomposition of square 5161's full series with period = 144 (one day).

**Trend component.** The trend is not flat: it rises gradually through November,
peaks in mid-December, and shows a sharp dip around Christmas (25–26 December),
consistent with reduced urban activity during the holiday period. This low-
amplitude trend does not threaten stationarity but is worth noting as a source
of non-stationarity that could affect models trained on early data and evaluated
in December.

**Seasonal component.** The seasonal component is stable and repeating,
confirming a consistent daily cycle throughout the observation period. The
amplitude of the seasonal component is large relative to the residual, indicating
that daily periodicity accounts for the majority of predictable variation in the
series.

**Residual component.** The residuals are mostly small but contain occasional
large spikes, corresponding to the anomalous bursts noted in Section 2.3. These
are not captured by either the trend or seasonal components and represent the
hardest part of the forecasting problem.

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

## References

Barlacchi, G., De Nadai, M., Larcher, R., Casella, A., Chitic, C., Torrisi, G.,
Antonelli, F., Vespignani, A., Pentland, A., & Lepri, B. (2015). A multi-source
dataset of urban life in the city of Milan and the Province of Trentino.
*Scientific Data*, 2, 150055. https://doi.org/10.1038/sdata.2015.55
