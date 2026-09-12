"""Baseline Benchmark Evaluation for Electricity Load Forecasting.

Evaluates Seasonal Naive 24-Hour and Seasonal Naive 7-Day baselines
on the 2014 Test Set for both:
1. Aggregate Grid Load (Day-ahead & Week-ahead)
2. Individual 370 Meters (Multi-series distribution)
"""

import numpy as np
import pandas as pd
from typing import Dict
from src.data_split import load_cleaned_data, get_aggregate_load, split_data
from src.metrics import evaluate_all, wmape, rmse, mae, nrmse


def evaluate_aggregate_baselines(data_full: pd.DataFrame) -> pd.DataFrame:
    """Evaluate baselines on the aggregate grid load for the 2014 test period."""
    print("\n" + "=" * 60)
    print("Evaluating Baselines on Aggregate Grid Load (2014 Test Year)")
    print("=" * 60)
    
    agg = get_aggregate_load(data_full)
    splits = split_data(agg)
    test_actual = splits["test"]
    
    results = {}
    
    # Baseline 1: Seasonal Naive 24h (lag 96 intervals)
    pred_sn24 = agg.shift(96).loc[test_actual.index]
    results["Seasonal Naive 24h (Day-Ahead)"] = evaluate_all(test_actual, pred_sn24)
    
    # Baseline 2: Seasonal Naive 7-Day (lag 672 intervals)
    pred_sn7d = agg.shift(672).loc[test_actual.index]
    results["Seasonal Naive 7-Day (Week-Ahead)"] = evaluate_all(test_actual, pred_sn7d)
    
    # Baseline 3: 4-Week Seasonal Mean (average of same 15-min slot over past 4 weeks)
    # Lags: 672, 1344, 2016, 2688
    lag_7 = agg.shift(672)
    lag_14 = agg.shift(1344)
    lag_21 = agg.shift(2016)
    lag_28 = agg.shift(2688)
    pred_4w = pd.concat([lag_7, lag_14, lag_21, lag_28], axis=1).mean(axis=1).loc[test_actual.index]
    results["4-Week Seasonal Average (Robust Baseline)"] = evaluate_all(test_actual, pred_4w)
    
    df_results = pd.DataFrame(results).T
    df_results["WMAPE (%)"] = df_results["WMAPE"] * 100
    df_results["NRMSE (%)"] = df_results["NRMSE"] * 100
    
    # Reorder columns
    cols = ["WMAPE (%)", "RMSE", "MAE", "NRMSE (%)"]
    df_results = df_results[cols]
    
    print(df_results.to_string(float_format=lambda x: f"{x:,.2f}"))
    return df_results


def evaluate_individual_meters_baselines(data_full: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Evaluate baselines across all 370 individual meters on the 2014 test period."""
    print("\n" + "=" * 60)
    print("Evaluating Baselines on 370 Individual Meters (2014 Test Year)")
    print("=" * 60)
    
    splits = split_data(data_full)
    test_actual = splits["test"]
    
    # Compute per-meter metrics
    metrics_sn24 = []
    metrics_sn7d = []
    
    pred_sn24 = data_full.shift(96).loc[test_actual.index]
    pred_sn7d = data_full.shift(672).loc[test_actual.index]
    
    for col in data_full.columns:
        yt = test_actual[col]
        # SN 24h
        yp24 = pred_sn24[col]
        m24 = evaluate_all(yt, yp24)
        m24["meter"] = col
        metrics_sn24.append(m24)
        
        # SN 7d
        yp7d = pred_sn7d[col]
        m7d = evaluate_all(yt, yp7d)
        m7d["meter"] = col
        metrics_sn7d.append(m7d)
        
    df_sn24 = pd.DataFrame(metrics_sn24).set_index("meter")
    df_sn7d = pd.DataFrame(metrics_sn7d).set_index("meter")
    
    summary = {}
    for name, df_m in [("Seasonal Naive 24h (Day-Ahead)", df_sn24), 
                       ("Seasonal Naive 7-Day (Week-Ahead)", df_sn7d)]:
        wmape_pct = df_m["WMAPE"] * 100
        nrmse_pct = df_m["NRMSE"] * 100
        summary[name] = {
            "Mean WMAPE (%)": wmape_pct.mean(),
            "Median WMAPE (%)": wmape_pct.median(),
            "25th Pct WMAPE (%)": wmape_pct.quantile(0.25),
            "75th Pct WMAPE (%)": wmape_pct.quantile(0.75),
            "Mean NRMSE (%)": nrmse_pct.mean(),
            "Median NRMSE (%)": nrmse_pct.median(),
            "Mean MAE (kW)": df_m["MAE"].mean(),
            "Mean RMSE (kW)": df_m["RMSE"].mean(),
        }
        
    df_summary = pd.DataFrame(summary).T
    print("\nSummary Across 370 Meters:")
    print(df_summary.to_string(float_format=lambda x: f"{x:,.2f}"))
    
    return {
        "summary": df_summary,
        "per_meter_sn24": df_sn24,
        "per_meter_sn7d": df_sn7d,
    }


def main():
    print("Loading cleaned dataset...")
    df = load_cleaned_data()
    print(f"Dataset shape: {df.shape}")
    
    # 1. Aggregate Load Baselines
    agg_results = evaluate_aggregate_baselines(df)
    
    # 2. Individual Meters Baselines
    meter_results = evaluate_individual_meters_baselines(df)
    
    # Save results summary to CSV for downstream reference
    agg_results.to_csv("data/baseline_aggregate_results.csv")
    meter_results["summary"].to_csv("data/baseline_individual_summary.csv")
    meter_results["per_meter_sn24"].to_csv("data/baseline_individual_sn24_per_meter.csv")
    meter_results["per_meter_sn7d"].to_csv("data/baseline_individual_sn7d_per_meter.csv")
    print("\nResults successfully saved to data/baseline_*.csv")


if __name__ == "__main__":
    main()
