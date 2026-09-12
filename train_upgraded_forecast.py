"""Upgraded Global LightGBM (v2) with Dynamic Rolling Baselines & Outage Resilience.

Resolves the 5 empirical shortcomings identified in v1:
1. Dynamic 14-day rolling baseline (fixes concept drift & level shifts like MT_332).
2. Cascading lag fallback (handles missing data and sensor outages like MT_348).
3. Top-down hierarchical grid ratio (supplies macro weather/system context).
4. Huber objective with peak protection (mitigates L1 median underestimation).
5. Comprehensive out-of-sample evaluation on all 370 meters over 2014.
"""

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data_split import load_cleaned_data, get_aggregate_load, split_data
from src.features import create_calendar_features, compute_dynamic_rolling_baseline, compute_cascading_lag
from src.metrics import evaluate_all, wmape

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


def train_upgraded_global_lightgbm(df_full: pd.DataFrame, agg_series: pd.Series, sample_size: int = 50):
    print("\n" + "=" * 75)
    print("Training Upgraded Global LightGBM (v2) with Dynamic Rolling Baselines")
    print("=" * 75)
    
    splits = split_data(df_full)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    
    # Precompute macro grid hierarchical signal (day-ahead available)
    # Ratio of macro load yesterday to 2 days ago (captures grid-wide surge/drop)
    agg_lag96 = agg_series.shift(96)
    agg_lag192 = agg_series.shift(192)
    agg_macro_ratio = (agg_lag96 / agg_lag192.replace(0, np.nan)).clip(0.5, 2.0).fillna(1.0)
    
    cal_feat = create_calendar_features(df_full.index)
    
    # Stratified sample of representative meters
    meter_means = train_df.mean()
    valid_meters = meter_means[meter_means > 1.0].index
    selected_meters = valid_meters[:: len(valid_meters) // sample_size][:sample_size]
    print(f"Training on {len(selected_meters)} representative meters...")
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    # Precompute dynamic rolling baselines for all selected meters
    for m in selected_meters:
        s = df_full[m]
        roll_mean, roll_std = compute_dynamic_rolling_baseline(s, min_lag=96, window=14 * 96)
        
        # Guard against zero or NaN baseline
        roll_mean = roll_mean.replace(0, np.nan).fillna(s.expanding().mean()).replace(0, 1.0)
        
        # Relative scaling: target is load divided by dynamic rolling baseline
        y_scaled = s / roll_mean
        
        # Cascading lag 96 (falls back to 192 or 672 during outages)
        lag96_cascaded, is_fallback = compute_cascading_lag(s, primary_lag=96, fallbacks=[192, 288, 672])
        lag672_cascaded, _ = compute_cascading_lag(s, primary_lag=672, fallbacks=[1344, 2016])
        
        scaled_lag96 = lag96_cascaded / roll_mean
        scaled_lag672 = lag672_cascaded / roll_mean
        scaled_diff = scaled_lag96 - scaled_lag672
        
        # Rolling stats on shifted series
        s_shifted = s.shift(96) / roll_mean
        roll_mean_4 = s_shifted.rolling(4).mean()
        roll_mean_96 = s_shifted.rolling(96).mean()
        
        # Feature matrix for this meter
        feat_m = pd.concat([
            cal_feat,
            pd.DataFrame({
                "scaled_lag_96": scaled_lag96,
                "scaled_lag_672": scaled_lag672,
                "scaled_diff": scaled_diff,
                "roll_mean_4": roll_mean_4,
                "roll_mean_96": roll_mean_96,
                "is_fallback": is_fallback,
                "macro_grid_ratio": agg_macro_ratio,
            }, index=df_full.index)
        ], axis=1)
        
        # Train slice (subsampled by 4 for efficiency)
        tr_m = feat_m.loc[train_df.index]
        y_tr_m = y_scaled.loc[train_df.index]
        mask_tr = tr_m.notna().all(axis=1) & y_tr_m.notna()
        X_train_list.append(tr_m[mask_tr].iloc[::4])
        y_train_list.append(y_tr_m[mask_tr].iloc[::4])
        
        # Val slice
        val_m = feat_m.loc[val_df.index]
        y_val_m = y_scaled.loc[val_df.index]
        mask_val = val_m.notna().all(axis=1) & y_val_m.notna()
        X_val_list.append(val_m[mask_val].iloc[::4])
        y_val_list.append(y_val_m[mask_val].iloc[::4])
        
    X_train = pd.concat(X_train_list, axis=0)
    y_train = pd.concat(y_train_list, axis=0)
    X_val = pd.concat(X_val_list, axis=0)
    y_val = pd.concat(y_val_list, axis=0)
    
    print(f"Dataset compiled: {len(X_train):,} training instances across {X_train.shape[1]} features.")
    
    # Train Upgraded LightGBM with Huber objective
    # Huber loss is quadratic for small errors and linear for large errors,
    # protecting against peak underestimation while remaining robust against outlier spikes
    print("Fitting Upgraded LightGBM (Huber objective)...")
    model_lgb = lgb.LGBMRegressor(
        objective="huber",
        huber_alpha=0.9,
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=7,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)]
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    print(f"Best iteration: {model_lgb.best_iteration_}")
    
    # Save model
    joblib.dump(model_lgb, "models/lgbm_global_v2_dynamic.joblib")
    print("Model saved to models/lgbm_global_v2_dynamic.joblib")
    
    # Evaluate across ALL 370 meters on 2014 Test Year!
    print("\nEvaluating Upgraded Model across all 370 individual meters on 2014 Test Year...")
    metrics_v2 = []
    
    test_idx = test_df.index
    cal_test = cal_feat.loc[test_idx]
    agg_macro_test = agg_macro_ratio.loc[test_idx]
    
    for col in df_full.columns:
        s = df_full[col]
        roll_mean, _ = compute_dynamic_rolling_baseline(s, min_lag=96, window=14 * 96)
        roll_mean_test = roll_mean.loc[test_idx].replace(0, np.nan).fillna(s.expanding().mean()).replace(0, 1.0)
        
        lag96_cascaded, is_fallback = compute_cascading_lag(s, primary_lag=96, fallbacks=[192, 288, 672])
        lag672_cascaded, _ = compute_cascading_lag(s, primary_lag=672, fallbacks=[1344, 2016])
        
        scaled_lag96 = lag96_cascaded.loc[test_idx] / roll_mean_test
        scaled_lag672 = lag672_cascaded.loc[test_idx] / roll_mean_test
        scaled_diff = scaled_lag96 - scaled_lag672
        
        s_shifted = s.shift(96) / roll_mean
        roll_mean_4 = s_shifted.rolling(4).mean().loc[test_idx]
        roll_mean_96 = s_shifted.rolling(96).mean().loc[test_idx]
        
        feat_test_m = pd.concat([
            cal_test,
            pd.DataFrame({
                "scaled_lag_96": scaled_lag96,
                "scaled_lag_672": scaled_lag672,
                "scaled_diff": scaled_diff,
                "roll_mean_4": roll_mean_4,
                "roll_mean_96": roll_mean_96,
                "is_fallback": is_fallback.loc[test_idx],
                "macro_grid_ratio": agg_macro_test,
            }, index=test_idx)
        ], axis=1)
        
        # Predict scaled ratio and multiply back by dynamic rolling baseline!
        scaled_preds = model_lgb.predict(feat_test_m)
        pred_kw = scaled_preds * roll_mean_test.values
        
        # Clip negative predictions to 0 (power load cannot be negative)
        pred_kw = np.clip(pred_kw, 0, None)
        
        yt_test = test_df[col].values
        m_eval = evaluate_all(yt_test, pred_kw)
        m_eval["meter"] = col
        metrics_v2.append(m_eval)
        
    df_metrics_v2 = pd.DataFrame(metrics_v2).set_index("meter")
    return df_metrics_v2


def summarize_metrics(df_metrics: pd.DataFrame, name: str) -> dict:
    w = df_metrics["WMAPE"] * 100
    n = df_metrics["NRMSE"] * 100
    return {
        "Model": name,
        "Mean WMAPE (%)": w.mean(),
        "Median WMAPE (%)": w.median(),
        "25th Pct WMAPE (%)": w.quantile(0.25),
        "75th Pct WMAPE (%)": w.quantile(0.75),
        "Mean NRMSE (%)": n.mean(),
        "Median NRMSE (%)": n.median(),
        "Mean MAE (kW)": df_metrics["MAE"].mean(),
        "Mean RMSE (kW)": df_metrics["RMSE"].mean(),
    }


def main():
    print("Loading cleaned dataset...")
    df = load_cleaned_data()
    agg = get_aggregate_load(df)
    
    # Train and evaluate upgraded v2 model
    df_v2 = train_upgraded_global_lightgbm(df, agg)
    df_v2.to_csv("data/model_individual_lgb_v2_dynamic_per_meter.csv")
    
    # Load prior results for direct comparison
    df_sn = pd.read_csv("data/baseline_individual_sn24_per_meter.csv", index_col="meter")
    df_v1 = pd.read_csv("data/model_individual_lgb_per_meter.csv", index_col="meter")
    df_ridge = pd.read_csv("data/model_individual_ridge_per_meter.csv", index_col="meter")
    
    # Comparison table
    comp = [
        summarize_metrics(df_sn, "1. Seasonal Naive 24h (Baseline)"),
        summarize_metrics(df_ridge, "2. Autoregressive Ridge (Per-Meter)"),
        summarize_metrics(df_v1, "3. Global LightGBM v1 (Static Normalization)"),
        summarize_metrics(df_v2, "4. Upgraded Global LightGBM v2 (Dynamic Rolling Baseline)")
    ]
    df_comp = pd.DataFrame(comp).set_index("Model")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK: All 370 Meters on 2014 Out-of-Sample Test Year")
    print("=" * 80)
    print(df_comp.to_string(float_format=lambda x: f"{x:,.2f}"))
    df_comp.to_csv("data/model_individual_meters_v1_vs_v2_comparison.csv")
    
    # Specific inspection of prior failure meters
    problem_meters = ["MT_332", "MT_348", "MT_093", "MT_066", "MT_130"]
    print("\n" + "=" * 80)
    print("DIAGNOSTIC VERIFICATION: Performance on Former Failure Case Meters")
    print("=" * 80)
    
    diag_rows = []
    for m in problem_meters:
        if m in df_v2.index and m in df_v1.index and m in df_sn.index:
            diag_rows.append({
                "Meter": m,
                "Seasonal Naive WMAPE (%)": df_sn.loc[m, "WMAPE"] * 100,
                "LightGBM v1 (Static) WMAPE (%)": df_v1.loc[m, "WMAPE"] * 100,
                "LightGBM v2 (Dynamic) WMAPE (%)": df_v2.loc[m, "WMAPE"] * 100,
                "v2 Improvement vs v1 (%)": (df_v1.loc[m, "WMAPE"] - df_v2.loc[m, "WMAPE"]) * 100
            })
    df_diag = pd.DataFrame(diag_rows).set_index("Meter")
    print(df_diag.to_string(float_format=lambda x: f"{x:,.2f}"))
    df_diag.to_csv("data/model_failure_cases_resolution_comparison.csv")


if __name__ == "__main__":
    main()
