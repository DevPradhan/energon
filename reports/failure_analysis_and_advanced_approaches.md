# Failure Mode Analysis & Advanced Methodology Roadmap

This document formalizes the empirical failure modes identified in the Global LightGBM v2 forecasting model and outlines the five advanced methodological architectures currently **UNDER TESTING** to establish production-grade reliability and probabilistic uncertainty quantification.

---

## 1. Empirical Failure Modes of Global LightGBM v2

Rigorous residual testing across all 370 meters over the 35,040 test intervals of 2014 revealed three structural limitations:

### A. High Residual Autocorrelation ($r \approx 0.82$ at Lag 1)
* **Finding**: The point forecast errors are not white noise. Residuals display strong temporal persistence:
  * Lag 1 (15m): $r \in [0.70, 0.86]$
  * Lag 2 (30m): $r \in [0.66, 0.77]$
  * Lag 4 (1h): $r \in [0.52, 0.67]$
* **Root Cause**: LightGBM performs tabular regression from $t-96$ without a continuous recurrent memory. When a consumer initiates an unforecasted industrial run, the model under-predicts repeatedly across consecutive intervals.

### B. Diurnal Error Heteroscedasticity (Daytime Doubling)
* **Finding**: Average WMAPE varies systematically across the 24-hour cycle:
  * **03:00–05:00 (Night Baseload)**: **$6.65\% - 7.09\%$** error.
  * **09:00–15:00 (Peak Workday Hours)**: **$13.59\% - 14.44\%$** error (error doubles).
  * **19:00–21:00 (Evening Peak)**: **$10.18\% - 11.84\%$** error.
* **Root Cause**: Nighttime consumption is dominated by deterministic baseload appliances. Daytime demand fluctuates with stochastic human behavior and production scheduling.

### C. The Bimodal / Intermittent Demand Trap (`MT_093`)
* **Finding**: `MT_093` exhibits an **$88.31\%$ WMAPE**. In the 2014 test period:
  * Median Load: **$3.55\text{ kW}$** (~85% of time in standby/idle)
  * 90th Percentile: **$191.37\text{ kW}$**
  * 95th Percentile: **$221.83\text{ kW}$** (surge machine operation)
* **Root Cause**: Continuous regression models predict a blended expectation ($\sim 30\text{ kW}$), which is incorrect in both physical states (too high when OFF, far too low when ON).

---

## 2. Advanced Methodologies [UNDER TESTING]

The following five paradigms are currently under formal investigation and benchmarking:

| Methodology | Status | Targeted Failure Mode | Core Innovation |
| :--- | :---: | :--- | :--- |
| **1. Empirical Stratified Residual Bootstrapping** | **[ACTIVE TESTING]** | Deterministic blindspots; unquantified peak risk | Stratified block-bootstrapping of heteroscedastic residuals to generate calibrated prediction intervals (80%, 90%, 95%) and reserve sizing. |
| **2. Two-Stage Hurdle / Intermittent Model** | **[UNDER TESTING]** | Bimodal/intermittent demand (`MT_093` 88% error) | Decouples binary equipment activation classification ($\mathbb{P}(\text{ON})$) from continuous conditional load regression. |
| **3. Hierarchical Forecast Reconciliation (MinT)** | **[UNDER TESTING]** | Grid incoherence; noisy bottom-tier meters | Uses the Minimum Trace theorem to reconcile the 3.26% aggregate forecast with bottom-level meter predictions via the summing matrix $\mathbf{S}$. |
| **4. Sequence & Deep State-Space Models (Seq2Seq LSTM / PatchTST)** | **[UNDER TESTING]** | High lag-1 residual autocorrelation ($r=0.82$) | End-to-end recurrent hidden state encoding past 7-day rate-of-change ($\frac{dy}{dt}$) and patch-level temporal attention. |
| **5. Hybrid GBDT + Autoregressive Residual Filter (AR-GBDT)** | **[UNDER TESTING]** | Systematic error momentum | Fits a fast autoregressive error filter on top of LightGBM predictions: $\hat{y}_t^* = \hat{y}_t^{\text{LGBM}} + \hat{e}_t^{\text{AR}}$. |

---

## 3. Active Initiative: Empirical Residual Bootstrapping Architecture

We initiate implementation with **Approach 1 (Empirical Residual Bootstrapping)**:

### Mathematical Formulation
For any forecast $\hat{y}_{m, t}$ produced by LightGBM, the probabilistic trajectory draw $b \in \{1, \dots, B\}$ is:
$$\hat{y}_{m, t}^{(b)} = \max\left(0, \hat{y}_{m, t} + e_{m, t}^{(b)}\right)$$

Where $e_{m, t}^{(b)}$ is sampled via **Diurnally Stratified Block Bootstrapping**:
1. Residual pool $\mathcal{E}_{m}(h, \text{is\_weekend})$ constructed from out-of-sample validation periods.
2. Contiguous blocks of length $L=4$ intervals (1 hour) are drawn to preserve intra-hour autocorrelation.
3. Quantiles $[\hat{q}_{0.05}, \hat{q}_{0.10}, \hat{q}_{0.50}, \hat{q}_{0.90}, \hat{q}_{0.95}]$ are computed for each timestamp.

### Success Metrics
* **Empirical Coverage**: Percentage of actual observations falling within $[L_{90\%}, U_{90\%}]$ (Target: $90.0\% \pm 2\%$).
* **Mean Prediction Interval Width (MPIW)**: Narrowness of the interval in kW.
* **Winkler Score**: Combined penalty for interval width and out-of-bounds violations.
