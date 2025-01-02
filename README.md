# ETF-Rotation-Based-on-Crowding-Strategy
ETF Rotation Based on Crowding Strategy

This plan is divided into five major modules: **Trading Volume**, **Turnover Rate**, **Volatility**, **Momentum**, and **Volume-Price Correlation**. Each module contains different types of indicators.

The **Trading Volume & Trading Volume to Market Value Ratio** module includes the following indicators: rolling average, rolling quantile, regression standardization, and regression quantile. The **Turnover Rate** module's indicators include rolling average, rolling quantile, rolling weighted average, and rolling deviation. The **Volatility** module includes closing price volatility and gap difference. The **Momentum** module's indicators are excess average, excess deviation, and excess volatility. In the **Volume-Price Correlation** module, we use the volume-price correlation coefficient and discrete correlation coefficients (-1, 0, 1).

Each rolling indicator's window period is set to 5 days, 10 days, 15 days, 20 days, 25 days, 30 days, 60 days, and 90 days. Each sub-indicator (by type and window) is tested for correlation with the maximum drawdown over the past 20 days, the more negative, the more effective.

Based on the construction and testing of the indicators mentioned earlier, the following key indicators are selected: `amount_rollq_60`, `proportion_rollq_15`, `turnover_rollq_90`, `turnover_deviation_30`, `volatility_20`, and `excess_volatility_30`.

The factor construction steps are as follows: First, normalize all indicators, then equally weight these indicators, and finally set the repositioning signal to operate on dates where the factor quantile is greater than the threshold=0.9.
