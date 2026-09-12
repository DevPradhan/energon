"""Multi-series forecasting across all 370 individual meters.

Implements:
1. Multi-output Autoregressive Ridge model across all 370 meters simultaneously.
2. Global Scale-Normalized LightGBM model with meter archetype/load factor static features.
3. Out-of-sample evaluation across all 370 individual meters on the 2014 test year (35,040 steps).
4. Direct comparison against the Seasonal Naive baseline hurdles.
"""

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.data_split import load_cleaned_data, split_data
from src.features import create_calendar_features
from src.metrics import evaluate_all, wmape

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


def build_meter_features_and_targets(df_full: pd.DataFrame, horizon_steps: int = 96):
    """Create calendar features and per-meter lag features for day-ahead (H=96) forecasting."""
    cal_feat = create_calendar_features(df_full.index)
    
    # Per-meter lag 96 (yesterday same time) and lag 672 (last week same time)
    lag_96 = df_full.shift(96)
    lag_672 = df_full.shift(672)
    diff_96_672 = lag_96 - lag_672
    
    return cal_feat, lag_96, lag_672, diff_96_672


def train_multi_output_ridge(df_full: pd.DataFrame):
    """Fit multi-output Ridge regression for all 370 meters simultaneously."""
    print("\n" + "=" * 70)
    print("Training Multi-Output Autoregressive Ridge Model (370 Meters)")
    print("=" * 70)
    
    cal_feat, lag_96, lag_672, diff_96_672 = build_meter_features_and_targets(df_full, horizon_steps=96)
    
    # Split data chronologically
    train_idx = split_data(df_full)["train"].index
    val_idx = split_data(df_full)["val"].index
    test_idx = split_data(df_full)["test"].index
    
    # For each meter, build feature matrix [cal_features, lag_96, lag_672, diff]
    # We can fit a shared calendar + autoregressive model per meter or vectorized
    y_train = df_full.loc[train_idx]
    y_test = df_full.loc[test_idx]
    
    # To handle all meters efficiently:
    # Model: y_{m, t} = f_m(Calendar) + alpha_m * lag_96_{m, t} + beta_m * lag_672_{m, t}
    # We fit a Ridge model per meter
    print("Fitting Ridge models across all 370 meters...")
    metrics_ridge = []
    preds_ridge = {}
    
    cal_train = cal_feat.loc[train_idx].values
    cal_test = cal_feat.loc[test_idx].values
    
    scaler = StandardScaler()
    cal_train_scaled = scaler.fit_transform(cal_train)
    cal_test_scaled = scaler.transform(cal_test)
    for col in df_full.columns:
        # Construct X for this meter
        yt_m = y_train[col].values
        l96_tr = lag_96.loc[train_idx, col].values
        l672_tr = lag_672.loc[train_idx, col].values
        
        # Filter valid rows
        mask_tr = np.isfinite(yt_m) & (np.isfinite(l96_tr) | np.isfinite(l672_tr))
        if mask_tr.sum() < 1000:
            continue
            
        X_tr_m = np.column_stack([cal_train_scaled[mask_tr], l96_tr[mask_tr], l672_tr[mask_tr]])
        y_tr_m = yt_m[mask_tr]
        
        reg_pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            Ridge(alpha=10.0)
        )
        reg_pipe.fit(X_tr_m, y_tr_m)
        
        # Test prediction
        l96_te = lag_96.loc[test_idx, col].values
        l672_te = lag_672.loc[test_idx, col].values
        
        X_te_m = np.column_stack([cal_test_scaled, l96_te, l672_te])
        pred_m = reg_pipe.predict(X_te_m)
        preds_ridge[col] = pred_m
        
        # Evaluate
        yt_test = y_test[col].values
        m_eval = evaluate_all(yt_test, pred_m)
        m_eval["meter"] = col
        metrics_ridge.append(m_eval)
        
    df_metrics = pd.DataFrame(metrics_ridge).set_index("meter")
    print(f"Evaluated Ridge on {len(df_metrics)} meters.")
    return df_metrics, pd.DataFrame(preds_ridge, index=test_idx)


def train_global_normalized_lightgbm(df_full: pd.DataFrame, sample_size: int = 50):
    """Train a Global Scale-Normalized LightGBM model.
    
    Each meter series is scaled by its training mean, allowing a single global GBDT
    to learn universal load dynamics across archetypes, then scaled back out-of-sample.
    """
    print("\n" + "=" * 70)
    print("Training Global Scale-Normalized LightGBM Model (Multi-Series)")
    print("=" * 70)
    
    splits = split_data(df_full)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    
    # Compute per-meter scale and static properties on train split
    meter_means = train_df.mean()
    meter_maxs = train_df.max()
    load_factors = meter_means / meter_maxs.replace(0, np.nan)
    
    # Select stratified sample of meters for training the global model
    # (covers residential, commercial, industrial)
    valid_meters = meter_means[meter_means > 1.0].index
    selected_meters = valid_meters[:: len(valid_meters) // sample_size][:sample_size]
    print(f"Training global model on {len(selected_meters)} representative meters...")
    
    cal_feat = create_calendar_features(df_full.index)
    
    # Build pooled training dataset
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    for m in selected_meters:
        scale = meter_means[m]
        if scale <= 0 or np.isnan(scale):
            continue
        lf = load_factors[m]
        
        # Normalized series
        norm_series = df_full[m] / scale
        
        lag_96 = norm_series.shift(96)
        lag_192 = norm_series.shift(192)
        lag_672 = norm_series.shift(672)
        diff_96_192 = lag_96 - lag_192
        diff_96_672 = lag_96 - lag_672
        roll_mean_4 = norm_series.shift(96).rolling(4).mean()
        roll_mean_96 = norm_series.shift(96).rolling(96).mean()
        
        feat_m = pd.concat([
            cal_feat,
            pd.DataFrame({
                "norm_lag_96": lag_96,
                "norm_lag_192": lag_192,
                "norm_lag_672": lag_672,
                "norm_diff_96_192": diff_96_192,
                "norm_diff_96_672": diff_96_672,
                "norm_roll_mean_4": roll_mean_4,
                "norm_roll_mean_96": roll_mean_96,
                "static_load_factor": lf,
            }, index=df_full.index)
        ], axis=1)
        
        # Train slice
        tr_m = feat_m.loc[train_df.index]
        y_tr_m = norm_series.loc[train_df.index]
        mask_tr = tr_m.notna().all(axis=1) & y_tr_m.notna()
        # Downsample rows by 4 (hourly resolution for training efficiency)
        X_train_list.append(tr_m[mask_tr].iloc[::4])
        y_train_list.append(y_tr_m[mask_tr].iloc[::4])
        
        # Val slice
        val_m = feat_m.loc[val_df.index]
        y_val_m = norm_series.loc[val_df.index]
        mask_val = val_m.notna().all(axis=1) & y_val_m.notna()
        X_val_list.append(val_m[mask_val].iloc[::4])
        y_val_list.append(y_val_m[mask_val].iloc[::4])
        
    X_train = pd.concat(X_train_list, axis=0)
    y_train = pd.concat(y_train_list, axis=0)
    X_val = pd.concat(X_val_list, axis=0)
    y_val = pd.concat(y_val_list, axis=0)
    
    print(f"Global dataset: {len(X_train):,} training instances across features {list(X_train.columns)}")
    
    # Train Global LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=600,
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
    print(f"Global LightGBM best iteration: {model_lgb.best_iteration_}")
    
    # Now evaluate on ALL 370 individual meters for the 2014 test set!
    print("\nEvaluating Global LightGBM across all 370 meters on 2014 Test Year...")
    metrics_lgb = []
    
    for col in df_full.columns:
        scale = meter_means[col]
        if scale <= 0 or np.isnan(scale):
            continue
        lf = load_factors[col]
        
        norm_series = df_full[col] / scale
        lag_96 = norm_series.shift(96)
        lag_192 = norm_series.shift(192)
        lag_672 = norm_series.shift(672)
        diff_96_192 = lag_96 - lag_192
        diff_96_672 = lag_96 - lag_672
        roll_mean_4 = norm_series.shift(96).rolling(4).mean()
        roll_mean_96 = norm_series.shift(96).rolling(96).mean()
        
        feat_test_m = pd.concat([
            cal_feat.loc[test_df.index],
            pd.DataFrame({
                "norm_lag_96": lag_96.loc[test_df.index],
                "norm_lag_192": lag_192.loc[test_df.index],
                "norm_lag_672": lag_672.loc[test_df.index],
                "norm_diff_96_192": diff_96_192.loc[test_df.index],
                "norm_diff_96_672": diff_96_672.loc[test_df.index],
                "norm_roll_mean_4": roll_mean_4.loc[test_df.index],
                "norm_roll_mean_96": roll_mean_96.loc[test_df.index],
                "static_load_factor": lf,
            }, index=test_df.index)
        ], axis=1)
        
        norm_pred = model_lgb.predict(feat_test_m)
        pred_kw = norm_pred * scale
        
        yt_test = test_df[col].values
        m_eval = evaluate_all(yt_test, pred_kw)
        m_eval["meter"] = col
        metrics_lgb.append(m_eval)
        
    df_metrics = pd.DataFrame(metrics_lgb).set_index("meter")
    
    # Save model
    joblib.dump(model_lgb, "models/lgbm_global_individual_meters.joblib")
    print("Global LightGBM model saved: models/lgbm_global_individual_meters.joblib")
    
    return df_metrics


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
    
    # 1. Load baseline results
    df_sn24 = pd.read_csv("data/baseline_individual_sn24_per_meter.csv", index_col="meter")
    
    # 2. Train Ridge
    df_ridge, _ = train_multi_output_ridge(df)
    
    # 3. Train Global LightGBM
    df_lgb = train_global_normalized_lightgbm(df)
    
    # Summary Comparison
    summaries = [
        summarize_metrics(df_sn24, "Benchmark: Seasonal Naive 24h"),
        summarize_metrics(df_ridge, "Autoregressive Ridge (Per-Meter)"),
        summarize_metrics(df_lgb, "Global Normalized LightGBM")
    ]
    df_summary = pd.DataFrame(summaries).set_index("Model")
    
    print("\n" + "=" * 70)
    print("Final Performance Comparison Across All 370 Meters (2014 Test Year)")
    print("=" * 70)
    print(df_summary.to_string(float_format=lambda x: f"{x:,.2f}"))
    
    df_summary.to_csv("data/model_individual_meters_comparison.csv")
    df_ridge.to_csv("data/model_individual_ridge_per_meter.csv")
    df_lgb.to_csv("data/model_individual_lgb_per_meter.csv")
    print("\nResults saved to data/model_individual_*.csv")


if __name__ == "__main__":
    main()
