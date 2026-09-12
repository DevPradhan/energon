"""Train and evaluate forecasting models on Aggregate Grid Load.

Covers:
- Day-Ahead (H=96) and Week-Ahead (H=672) horizons.
- Models: Seasonal Naive, Ridge Regression, LightGBM Regressor.
- Out-of-sample evaluation on full 2014 test year (35,040 steps).
- Feature importance extraction and visual forecast comparisons.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_split import load_cleaned_data, get_aggregate_load, split_data
from src.features import create_aggregate_features
from src.metrics import evaluate_all

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


def train_and_eval_horizon(agg_series: pd.Series, horizon: str = "day_ahead") -> dict:
    horizon_label = "Day-Ahead (H=96 / 24h)" if horizon == "day_ahead" else "Week-Ahead (H=672 / 7d)"
    print("\n" + "=" * 70)
    print(f"Training & Evaluating Models for: {horizon_label}")
    print("=" * 70)
    
    # 1. Feature Extraction
    X, y = create_aggregate_features(agg_series, horizon=horizon)
    
    # 2. Chronological Splitting
    X_splits = split_data(X)
    y_splits = split_data(y)
    
    X_train, y_train = X_splits["train"], y_splits["train"]
    X_val, y_val = X_splits["val"], y_splits["val"]
    X_test, y_test = X_splits["test"], y_splits["test"]
    
    # Drop rows with NaNs (due to initial lag shifting)
    train_mask = X_train.notna().all(axis=1) & y_train.notna()
    val_mask = X_val.notna().all(axis=1) & y_val.notna()
    test_mask = X_test.notna().all(axis=1) & y_test.notna()
    
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    print(f"Train samples: {len(X_train):,}, Val samples: {len(X_val):,}, Test samples: {len(X_test):,}")
    print(f"Features: {list(X_train.columns)}")
    
    results = {}
    test_preds = {}
    
    # Model 0: Baseline Benchmark (Seasonal Naive)
    if horizon == "day_ahead":
        baseline_pred = X_test["lag_96"]
        baseline_name = "Benchmark: Seasonal Naive 24h"
    else:
        baseline_pred = X_test["lag_672"]
        baseline_name = "Benchmark: Seasonal Naive 7-Day"
        
    results[baseline_name] = evaluate_all(y_test, baseline_pred)
    test_preds[baseline_name] = baseline_pred
    
    # Model 1: Regularized Linear Baseline (Ridge)
    print("\nFitting Ridge Regression...")
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=100.0))
    ])
    ridge_pipeline.fit(X_train, y_train)
    pred_ridge = ridge_pipeline.predict(X_test)
    results["Ridge Regression"] = evaluate_all(y_test, pred_ridge)
    test_preds["Ridge"] = pred_ridge
    
    # Model 2: LightGBM Regressor (Tuned for time-series forecasting)
    print("Fitting LightGBM Regressor...")
    model_lgb = lgb.LGBMRegressor(
        objective="regression_l1",  # Optimizes MAE directly (aligns with WMAPE)
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    
    best_iter = model_lgb.best_iteration_
    print(f"LightGBM early stopping best iteration: {best_iter}")
    
    pred_lgb = model_lgb.predict(X_test)
    results["LightGBM Regressor"] = evaluate_all(y_test, pred_lgb)
    test_preds["LightGBM"] = pred_lgb
    
    # Format comparison table
    df_res = pd.DataFrame(results).T
    df_res["WMAPE (%)"] = df_res["WMAPE"] * 100
    df_res["NRMSE (%)"] = df_res["NRMSE"] * 100
    cols = ["WMAPE (%)", "RMSE", "MAE", "NRMSE (%)"]
    df_res = df_res[cols]
    
    print("\nOut-of-Sample Performance Comparison (2014 Test Set):")
    print(df_res.to_string(float_format=lambda x: f"{x:,.2f}"))
    
    # Save Model Artifact
    model_file = f"models/lgbm_aggregate_{horizon}.joblib"
    joblib.dump(model_lgb, model_file)
    print(f"Model saved: {model_file}")
    
    # Feature Importances
    fi = pd.Series(model_lgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    
    return {
        "results_table": df_res,
        "feature_importances": fi,
        "test_actual": y_test,
        "test_preds": test_preds,
        "model": model_lgb,
        "horizon": horizon
    }


def plot_results(day_res: dict, week_res: dict):
    """Plot feature importances and forecast visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # 1. Feature Importance: Day-Ahead
    fi_day = day_res["feature_importances"].head(10)
    axes[0, 0].barh(fi_day.index[::-1], fi_day.values[::-1], color="#2563eb")
    axes[0, 0].set_title("Top 10 Features: Day-Ahead Forecast (H=96)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Feature Importance (Split)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Feature Importance: Week-Ahead
    fi_week = week_res["feature_importances"].head(10)
    axes[0, 1].barh(fi_week.index[::-1], fi_week.values[::-1], color="#7c3aed")
    axes[0, 1].set_title("Top 10 Features: Week-Ahead Forecast (H=672)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Feature Importance (Split)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Forecast vs Actual: Representative 2-week slice in 2014 Test Year
    # Slice: 2014-06-01 to 2014-06-14 (Summer sample)
    slice_range = slice("2014-06-02 00:00:00", "2014-06-15 23:45:00")
    y_test = day_res["test_actual"].loc[slice_range]
    lgb_day_pred = pd.Series(day_res["test_preds"]["LightGBM"], index=day_res["test_actual"].index).loc[slice_range]
    sn_day_pred = day_res["test_preds"]["Benchmark: Seasonal Naive 24h"].loc[slice_range]
    
    axes[1, 0].plot(y_test.index, y_test.values, color="black", linewidth=1.8, label="Actual Grid Load")
    axes[1, 0].plot(lgb_day_pred.index, lgb_day_pred.values, color="#2563eb", linewidth=1.5, label="LightGBM Day-Ahead")
    axes[1, 0].plot(sn_day_pred.index, sn_day_pred.values, color="#dc2626", linestyle=":", alpha=0.6, label="Seasonal Naive 24h")
    axes[1, 0].set_title("Day-Ahead Forecast vs Actual (2014 Summer Sample)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Aggregate Load (kW)")
    axes[1, 0].legend(loc="upper right", frameon=True)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis="x", rotation=25)
    
    # 4. Week-Ahead Forecast vs Actual
    lgb_week_pred = pd.Series(week_res["test_preds"]["LightGBM"], index=week_res["test_actual"].index).loc[slice_range]
    sn_week_pred = week_res["test_preds"]["Benchmark: Seasonal Naive 7-Day"].loc[slice_range]
    
    axes[1, 1].plot(y_test.index, y_test.values, color="black", linewidth=1.8, label="Actual Grid Load")
    axes[1, 1].plot(lgb_week_pred.index, lgb_week_pred.values, color="#7c3aed", linewidth=1.5, label="LightGBM Week-Ahead")
    axes[1, 1].plot(sn_week_pred.index, sn_week_pred.values, color="#dc2626", linestyle=":", alpha=0.6, label="Seasonal Naive 7-Day")
    axes[1, 1].set_title("Week-Ahead Forecast vs Actual (2014 Summer Sample)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Aggregate Load (kW)")
    axes[1, 1].legend(loc="upper right", frameon=True)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=25)
    
    plt.tight_layout()
    fig_file = "reports/figures/model_aggregate_forecast_results.png"
    plt.savefig(fig_file, bbox_inches="tight")
    plt.close()
    print(f"Forecast comparison plot saved: {fig_file}")


def main():
    print("Loading cleaned dataset...")
    df = load_cleaned_data()
    agg = get_aggregate_load(df)
    
    # Day-Ahead
    day_res = train_and_eval_horizon(agg, horizon="day_ahead")
    
    # Week-Ahead
    week_res = train_and_eval_horizon(agg, horizon="week_ahead")
    
    # Plot results
    plot_results(day_res, week_res)
    
    # Save combined results
    combined_summary = pd.concat({
        "Day-Ahead (24h)": day_res["results_table"],
        "Week-Ahead (7d)": week_res["results_table"]
    })
    combined_summary.to_csv("data/model_aggregate_forecast_results.csv")
    print("\nSummary saved to data/model_aggregate_forecast_results.csv")


if __name__ == "__main__":
    main()
