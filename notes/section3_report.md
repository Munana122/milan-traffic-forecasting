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
temporal dependencies without manual feature engineering. The original LSTM
architecture was introduced by Hochreiter & Schmidhuber (1997) specifically to
address the vanishing gradient problem in standard RNNs, enabling the model to
retain information across hundreds of time steps. This property makes LSTM
well-suited to traffic series with strong multi-lag autocorrelation. A broader
body of work on time-series forecasting confirms that LSTM consistently
outperforms plain RNNs on sequential prediction tasks, while noting that
training cost scales with sequence length and hidden dimension — a relevant
trade-off when comparing models empirically (Zhang, 2003).

**Hybrid approaches.** Zhang (2003) demonstrated that combining ARIMA with a
neural network — where ARIMA captures the linear structure and the neural
component models the nonlinear residuals — outperformed either model alone on
several benchmark series. This finding implies that real-world traffic series
contain both a strong periodic/linear component and a nonlinear residual that
purely statistical models leave unexplained, motivating the inclusion of both
SARIMA and LSTM in this study.

**Feature-based machine learning.** Gradient boosting methods (XGBoost, LightGBM)
reframe forecasting as supervised regression over engineered lag and calendar
features. They train faster than deep models, require no sequence architecture,
and provide feature-importance interpretability. They have shown competitive
accuracy on many tabular forecasting benchmarks, though their performance depends
heavily on the quality of the hand-crafted features (Chen & Guestrin, 2016).

## 3.2 Findings from exploratory analysis

The EDA in Section 2 produced three results that directly inform model selection:

1. **Strong stationarity.** The Augmented Dickey-Fuller test on square 5161
   (the highest-traffic cell) returned ADF = −19.03, p ≈ 0. The null hypothesis
   of a unit root is rejected with overwhelming confidence: the series is
   stationary in levels and requires no differencing.

2. **Pronounced daily periodicity.** The ACF plot showed sharp, regularly-spaced
   peaks at lags 144, 288, and 432 (corresponding to 1, 2, and 3 days at
   10-minute resolution), confirming a strong daily seasonal cycle. The seasonal
   decomposition (period = 144) isolated a clean, stable seasonal component
   alongside a slowly varying trend.

3. **Heavy-tailed traffic distribution.** Total traffic across the 10,000 squares
   is strongly right-skewed (mean 555,289, median 277,871, max 12,740,060).
   Square 5161 carries roughly 23× the mean load, making it a demanding test
   case for any model.

## 3.3 Selected models

Three models were selected to span statistical, deep-sequential, and
feature-based-ML paradigms — architecturally distinct enough to support a
meaningful comparison.

### Model 1: SARIMA

SARIMA (Seasonal AutoRegressive Integrated Moving Average) is selected as the
classical statistical baseline. Its seasonal order is set to period s = 144
(one day), directly motivated by the ACF peaks at lags 144, 288, and 432
identified in Section 2. Because the ADF test confirmed stationarity in levels,
the non-seasonal differencing order d can be set to 0, simplifying the model.
SARIMA is fully interpretable — each parameter has a direct statistical meaning
— and its forecasts are fast to generate at inference time. Its principal
limitation is the linearity assumption: it cannot represent the nonlinear,
bursty dynamics visible in the high-traffic time series, and fitting a
full-order seasonal model to 89 M rows requires careful order selection to
avoid overfitting (Box et al., 2015).

### Model 2: LSTM

LSTM is selected as the primary deep-learning model. LSTM's gating mechanism
(Hochreiter & Schmidhuber, 1997) allows it to selectively retain information
across hundreds of time steps, making it well-suited to the long-lag
autocorrelation structure observed in the ACF. Unlike SARIMA, LSTM can model
nonlinear interactions between past values without any explicit seasonal
parameterisation. Its weaknesses are training cost (scales with sequence length
and hidden dimension), sensitivity to hyperparameter choices, and limited
interpretability.

### Model 3: XGBoost with lag features

XGBoost is selected as a non-recurrent, feature-based ML baseline. The
forecasting problem is recast as supervised regression: for each time step t,
the input is a feature vector of lagged traffic values (e.g. lags 1–12 for
recent context, lag 144 for yesterday's same interval, lag 288 for two days
prior) plus calendar features (hour of day, day of week). This design directly
encodes the daily periodicity identified in the ACF without requiring the model
to learn it implicitly. XGBoost trains in seconds rather than minutes, provides
feature-importance scores that reveal which lags are most predictive, and is
robust to outliers through its gradient-boosting objective. Its core limitation
is that it treats each prediction independently — it has no native notion of
sequence continuity — so its accuracy depends entirely on how well the lag
features capture the relevant temporal structure (Chen & Guestrin, 2016).

## 3.4 Summary comparison

| | SARIMA | LSTM | XGBoost |
|---|---|---|---|
| Paradigm | Statistical | Deep sequential | Feature-based ML |
| Handles nonlinearity | No | Yes | Yes |
| Explicit seasonality | Yes (s=144) | Learned | Via lag features |
| Training speed | Fast | Slow | Very fast |
| Interpretability | High | Low | Medium |
| Stationarity required | No (d=0 here) | No | No |
| Key motivation | ACF/ADF baseline | Hochreiter & Schmidhuber (1997) | Speed + interpretability |

## References

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
*Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
*Proceedings of KDD 2016*, 785–794. https://doi.org/10.1145/2939672.2939785

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Trinh, H. D., Giupponi, L., & Dini, P. (2018). Mobile traffic forecasting using
LSTM-based prediction models. *Proceedings of the 21st ACM International
Conference on Modelling, Analysis and Simulation of Wireless and Mobile Systems
(MSWiM ’18)*, 223–230. https://doi.org/10.1145/3242102.3242125

Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural
network model. *Neurocomputing*, 50, 159–175.
https://doi.org/10.1016/S0925-2312(01)00702-0
