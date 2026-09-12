# Energon ⚡

High-precision electricity load forecasting pipeline and evaluation benchmark for smart grid telemetry.

---

## 📌 Project Overview

**Energon** is an end-to-end framework designed to process, clean, evaluate, and forecast multi-client electricity consumption at high temporal frequency (15-minute resolution).

* **Dataset**: 370 client electricity meters (`MT_001` to `MT_370`) recorded at 15-minute intervals from 2011 through 2014 (140,256 timesteps per series).
* **Targets**:
  1. **Aggregate Grid Load**: System-wide total electricity demand across all active meters.
  2. **Individual Meters**: Multi-series forecasting for individual consumers (industrial, commercial, and residential).
* **Forecasting Horizons**:
  - **Day-Ahead**: 96 intervals (24 hours).
  - **Week-Ahead**: 672 intervals (7 days).

---

## 🧹 Data Cleaning & Preprocessing

The raw telemetry data exhibits intermittent sensor drops and delayed customer onboarding:
1. **Pre-Online Inactivity**: Leading zeros prior to a meter's initial connection are converted to `NaN` to prevent zero-inflation.
2. **Short Blips (≤ 1 hour / 4 steps)**: Linearly interpolated across continuous load trajectories.
3. **Medium Outages (1 to 24 hours)**: Seasonally imputed using matching 24-hour diurnal patterns.
4. **Extended Outages (> 24 hours)**: Converted to `NaN` to prevent hallucinating artificial consumption.
5. **Storage**: Exported to highly compressed, strictly-typed Parquet format (`data/cleaned_electricity_load.parquet`).

---

## 🧪 Evaluation Framework

### Chronological Splits (Zero Data Leakage)
* **Train**: `2012-01-01` to `2013-09-30` (61,343 steps)
* **Validation**: `2013-10-01` to `2013-12-31` (8,832 steps, Q4 2013)
* **Test**: `2014-01-01` to `2014-12-31` (35,040 steps, full year 2014)

### Primary Metrics
* **WMAPE (Weighted MAPE / Normalized MAE)**: Primary scale-free metric:
  $$\text{WMAPE} = \frac{\sum |y_t - \hat{y}_t|}{\sum y_t}$$
* **RMSE & MAE**: Measures absolute errors in kilowatt (kW) scale.
* **NRMSE**: Scale-normalized RMSE for cross-meter comparison.

---

## 📊 Forecasting Model Benchmark Results (2014 Test Year)

Out-of-sample evaluation on all 35,040 test steps in 2014 across both targets:

### 1. Aggregate Grid Load (System-Wide Demand)
| Horizon | Model | WMAPE (%) | RMSE (kW) | MAE (kW) | NRMSE (%) | vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Day-Ahead (24h / 96 steps)** | **Seasonal Naive 24h** | 3.82% | 13,931.13 | 8,571.56 | 6.20% | Baseline |
| | **Autoregressive Ridge** | 3.55% | 12,526.70 | 7,974.24 | 5.58% | +7.1% |
| | **LightGBM Regressor** | **3.26%** | **11,353.96** | **7,316.04** | **5.06%** | **+14.7%** |
| **Week-Ahead (7d / 672 steps)**| **Seasonal Naive 7-Day**| 5.77% | 21,211.46 | 12,961.39 | 9.45% | Baseline |
| | **Autoregressive Ridge** | 5.90% | 19,462.57 | 13,242.77 | 8.67% | Mixed |
| | **LightGBM Regressor** | **5.67%** | **20,104.78** | **12,739.49** | **8.95%** | **Beats Baseline** |

### 2. Individual 370 Meters (Multi-Series Consumer Demand)
| Model | Median WMAPE (%) | Mean WMAPE (%) | 25th Pct | 75th Pct | Median NRMSE (%) | Mean MAE (kW) | Mean RMSE (kW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive 24h** | 9.46% | 12.62% | 7.32% | 13.88% | 14.09% | 56.58 kW | 89.59 kW |
| **Autoregressive Ridge** | 8.23% | 11.03% | 6.68% | 11.80% | 12.04% | 55.03 kW | **81.29 kW** |
| **Global Normalized LightGBM** | **7.89%** | **10.12%** | **6.27%** | **11.17%** | **11.45%** | **53.52 kW** | 82.76 kW |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/DevPradhan/energon.git
cd energon
pip install -r requirements.txt
```

### 2. Run Data Cleaning & Baselines
```bash
python clean_electricity_data.py
python run_baselines.py
```

### 3. Train & Evaluate Forecasting Models
```bash
# Train Aggregate Grid Load Models (Day-Ahead & Week-Ahead)
python train_aggregate_models.py

# Train Multi-Series Individual Meters Models (Ridge & Global LightGBM)
python train_individual_meters_forecast.py
```

---

## 📁 Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── metrics.py                        # Vectorized WMAPE, RMSE, MAE, NRMSE
│   ├── data_split.py                     # Strict temporal splitting & aggregation
│   └── features.py                       # Calendar, multi-scale lag & rolling stats
├── train_aggregate_models.py             # Trains & benchmarks Ridge + LightGBM on Aggregate Load
├── train_individual_meters_forecast.py   # Multi-series Ridge & Global LightGBM across 370 meters
├── run_baselines.py                      # 2014 out-of-sample baseline runner
├── run_thorough_eda.py                   # Automated EDA visual analysis & report generator
├── explore_data.ipynb                    # Interactive Jupyter notebook for exploration
├── models/
│   ├── lgbm_aggregate_day_ahead.joblib   # Trained Day-Ahead aggregate model
│   ├── lgbm_aggregate_week_ahead.joblib  # Trained Week-Ahead aggregate model
│   └── lgbm_global_individual_meters.joblib # Global multi-series model
├── data/
│   ├── baseline_*.csv                    # Benchmark results across meters and aggregate
│   └── model_*.csv                       # Trained model evaluation results
├── requirements.txt
├── .gitignore
└── README.md
```
