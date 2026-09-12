# Diagnostic Review: Empirical Shortcomings of Global Normalized LightGBM (v1)

This report provides an empirical post-mortem of the initial Global Normalized LightGBM forecasting model evaluated on 370 electricity meters over the 2014 out-of-sample test period.

---

## 1. Summary of Empirical Diagnostics

* **Total Evaluated Nodes**: 347 valid continuous meters in 2014.
* **Beats Baseline**: The model outperformed the Seasonal Naive 24h baseline on **90.2% of meters** (313 out of 347).
* **Failure Cases**:
  * On **9.8% of meters (34 meters)**, the model performed worse than naive persistence.
  * On **2.9% of meters (10 meters)**, WMAPE degraded by more than **2.0 percentage points**.
  * On **6.3% of meters (22 meters)**, WMAPE exceeded **20%**.

---

## 2. In-Depth Root Cause Analysis & Empirical Evidence

### A. The Static Normalization Trap (Level Shift / Concept Drift)
* **Root Cause**: Dividing by the historical training mean $\mu_{\text{train}}$ assumes constant multi-year consumption scales.
* **Empirical Proof (`MT_332`)**:
  * Train Mean (2012–2013): **19.86 kW**
  * Test Mean (2014): **75.12 kW** (**+278.3% structural surge**)
  * Model Error: **48.08% WMAPE**
  * Baseline Error: **34.86% WMAPE** (Seasonal Naive adapted immediately to yesterday's 75 kW).
* **Fix**: Use a strictly causal rolling baseline $\mu_{t-96, \text{14d}}$ or relative adjustment modeling ($\frac{y_t}{y_{t-96}}$).

### B. Variance Asymmetry Across Heterogeneous Consumers
* **Root Cause**: Mean normalization ($\tilde{y} = y / \mu$) normalizes the mean to 1.0, but leaves the Coefficient of Variation ($CV = \sigma / \mu$) untouched.
* **Empirical Proof**:
  * Low-variance industrial meters have $CV \approx 0.10$.
  * Peaky intermittent meters like `MT_093` have $CV = 2.08$.
  * In a shared objective function, high-variance meters generate disproportionately large gradients, biasing tree splits away from steady baseload consumers.
* **Fix**: Full $Z$-score standardization ($z = \frac{y - \mu}{\sigma}$) or archetype-based sub-models.

### C. $L_1$ Loss Conditional Median Bias
* **Root Cause**: `objective="regression_l1"` targets the conditional median. In right-skewed load profiles, $\text{Median} < \text{Mean}$, causing systematic under-forecasting during extreme peak hours.
* **Fix**: Asymmetric Huber or Quantile Loss ($\tau \in [0.55, 0.60]$) to protect peak demand reliability.

### D. Information Isolation & Lack of Weather/Macro Signals
* **Root Cause**: Each meter model operated independently without regional weather or grid-level feedback.
* **Fix**: Hierarchical Top-Down Signal Injection using the 3.26% WMAPE Aggregate Grid Load forecast as an exogenous explanatory input.

### E. Missing Value Degradation During Sensor Outages
* **Root Cause**: During sensor outages lasting $>24$ hours (e.g. `MT_348` and `MT_130` with ~45% missing values in 2014), autoregressive lags ($t-96$, $t-672$) evaluate to `NaN`, forcing the model down default calendar split paths.
* **Fix**: Cascading lag fallback ($t-96 \to t-192 \to t-288 \to t-672$).
