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

## 📊 Baseline Benchmark Results (2014 Test Year)

Lower-bound baselines computed over 35,040 out-of-sample intervals in 2014:

### Aggregate Grid Load
| Model | Horizon | WMAPE (%) | RMSE (kW) | MAE (kW) | NRMSE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Seasonal Naive 24h** ($y_{t-96}$) | Day-Ahead (24h) | **3.82%** | 13,931.13 | 8,571.56 | 6.20% |
| **Seasonal Naive 7-Day** ($y_{t-672}$) | Week-Ahead (7d) | **5.77%** | 21,211.46 | 12,961.39 | 9.45% |
| **4-Week Seasonal Average** | Multi-week smoothed | **5.94%** | 21,176.29 | 13,332.30 | 9.43% |

### Individual 370 Meters (Summary Distribution)
| Metric Across All 370 Meters | Seasonal Naive 24h (Day-Ahead) | Seasonal Naive 7-Day (Week-Ahead) |
| :--- | :---: | :---: |
| **Mean WMAPE (%)** | **12.62%** | **12.64%** |
| **Median WMAPE (%)** | **9.46%** | **10.31%** |
| **25th – 75th Percentile WMAPE** | 7.32% – 13.88% | 8.35% – 13.84% |
| **Mean NRMSE (%)** | 20.95% | 20.44% |
| **Mean RMSE (kW)** | 89.59 kW | 105.74 kW |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/DevPradhan/energon.git
cd energon
pip install -r requirements.txt
```

### 2. Clean Data
```bash
python clean_electricity_data.py
```

### 3. Run Baseline Evaluation
```bash
python run_baselines.py
```

---

## 📁 Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── metrics.py           # Vectorized WMAPE, RMSE, MAE, NRMSE
│   └── data_split.py        # Strict temporal splitting & aggregation
├── clean_electricity_data.py # Automated tiered imputation pipeline
├── run_baselines.py         # 2014 out-of-sample benchmark runner
├── explore_data.ipynb       # Exploratory analysis notebook
├── data/
│   └── baseline_*.csv       # Benchmark results across meters and aggregate
├── requirements.txt
├── .gitignore
└── README.md
```
