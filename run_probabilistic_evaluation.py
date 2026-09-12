"""Run and benchmark Empirical Stratified Residual Bootstrapping.

Produces:
1. Probabilistic prediction intervals (80%, 90%, 95%) on Aggregate Load (2014 test set).
2. Probabilistic evaluation on 4 Consumer Archetypes (Residential, Commercial, Industrial, Intermittent).
3. Coverage (PICP), Sharpness (MPIW), and Winkler Scores.
4. Visual Probabilistic Fan-Chart figure for risk dispatch analysis.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_split import load_cleaned_data, get_aggregate_load, split_data
from src.features import create_aggregate_features, create_calendar_features, compute_dynamic_rolling_baseline, compute_cascading_lag
from src.bootstrapping import (
    build_stratified_residual_pool,
    generate_stratified_block_bootstrap,
    extract_prediction_intervals,
    evaluate_probabilistic_coverage
)

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data", exist_ok=True)


def evaluate_aggregate_probabilistic():
    print("\n" + "=" * 75)
    print("Evaluating Probabilistic Bootstrapping on Aggregate Grid Load (2014 Test Year)")
    print("=" * 75)
    
    df = load_cleaned_data()
    agg = get_aggregate_load(df)
    
    # Extract features for aggregate day-ahead
    X, y = create_aggregate_features(agg, horizon="day_ahead")
    splits_X = split_data(X)
    splits_y = split_data(y)
    
    # Validation split for building the residual pool (Q4 2013)
    val_mask = splits_X["val"].notna().all(axis=1) & splits_y["val"].notna()
    X_val = splits_X["val"][val_mask]
    y_val = splits_y["val"][val_mask]
    
    # Test split (2014 full year)
    test_mask = splits_X["test"].notna().all(axis=1) & splits_y["test"].notna()
    X_test = splits_X["test"][test_mask]
    y_test = splits_y["test"][test_mask]
    
    # Load trained model
    model = joblib.load("models/lgbm_aggregate_day_ahead.joblib")
    
    # Predict on validation to construct empirical residual pool
    pred_val = pd.Series(model.predict(X_val), index=y_val.index)
    residual_pool = build_stratified_residual_pool(y_val, pred_val, block_size=4)
    print(f"Constructed residual pool with {sum(len(b) for b in residual_pool.values())} contiguous 1-hour blocks.")
    
    # Predict on test set
    pred_test = pd.Series(model.predict(X_test), index=y_test.index)
    
    # Run Stratified Block Bootstrapping (500 simulated paths)
    print("Simulating 500 bootstrap paths across 35,040 test timestamps...")
    simulations = generate_stratified_block_bootstrap(
        pred_test, residual_pool, n_bootstraps=500, block_size=4, random_seed=42
    )
    
    # Extract intervals
    quantiles = extract_prediction_intervals(simulations, quantiles=[0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975])
    
    # Evaluate calibration
    eval_80 = evaluate_probabilistic_coverage(y_test, quantiles["q_10"], quantiles["q_90"], nominal_coverage=0.80)
    eval_90 = evaluate_probabilistic_coverage(y_test, quantiles["q_05"], quantiles["q_95"], nominal_coverage=0.90)
    eval_95 = evaluate_probabilistic_coverage(y_test, quantiles["q_02"], quantiles["q_97"], nominal_coverage=0.95)
    
    df_eval = pd.DataFrame([eval_80, eval_90, eval_95]).set_index("Nominal Target (%)")
    print("\nEmpirical Coverage & Calibration Metrics (Aggregate Load 2014):")
    print(df_eval.to_string(float_format=lambda x: f"{x:,.2f}"))
    
    return {
        "y_test": y_test,
        "pred_test": pred_test,
        "quantiles": quantiles,
        "eval_table": df_eval,
        "simulations": simulations
    }


def evaluate_archetype_probabilistic(df_full: pd.DataFrame, agg_series: pd.Series):
    print("\n" + "=" * 75)
    print("Evaluating Probabilistic Bootstrapping Across 4 Consumer Archetypes")
    print("=" * 75)
    
    archetypes = {
        "Commercial Office": "MT_333",
        "Residential Consumer": "MT_222",
        "Industrial Plant": "MT_081",
        "Intermittent Load": "MT_093"
    }
    
    splits = split_data(df_full)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    
    cal_feat = create_calendar_features(df_full.index)
    agg_lag96 = agg_series.shift(96)
    agg_lag192 = agg_series.shift(192)
    agg_macro_ratio = (agg_lag96 / agg_lag192.replace(0, np.nan)).clip(0.5, 2.0).fillna(1.0)
    
    model = joblib.load("models/lgbm_global_v2_dynamic.joblib")
    
    archetype_results = {}
    
    for name, meter_id in archetypes.items():
        s = df_full[meter_id]
        roll_mean, _ = compute_dynamic_rolling_baseline(s, min_lag=96, window=14 * 96)
        lag96_cascaded, is_fallback = compute_cascading_lag(s, primary_lag=96, fallbacks=[192, 288, 672])
        lag672_cascaded, _ = compute_cascading_lag(s, primary_lag=672, fallbacks=[1344, 2016])
        
        # 1. Validation predictions to build residual pool
        val_idx = val_df.index
        rm_val = roll_mean.loc[val_idx].replace(0, np.nan).fillna(s.expanding().mean()).replace(0, 1.0)
        
        feat_val = pd.concat([
            cal_feat.loc[val_idx],
            pd.DataFrame({
                "scaled_lag_96": lag96_cascaded.loc[val_idx] / rm_val,
                "scaled_lag_672": lag672_cascaded.loc[val_idx] / rm_val,
                "scaled_diff": (lag96_cascaded.loc[val_idx] - lag672_cascaded.loc[val_idx]) / rm_val,
                "roll_mean_4": (s.shift(96) / roll_mean).rolling(4).mean().loc[val_idx],
                "roll_mean_96": (s.shift(96) / roll_mean).rolling(96).mean().loc[val_idx],
                "is_fallback": is_fallback.loc[val_idx],
                "macro_grid_ratio": agg_macro_ratio.loc[val_idx],
            }, index=val_idx)
        ], axis=1)
        
        val_point_pred = pd.Series(
            np.clip(model.predict(feat_val) * rm_val.values, 0, None),
            index=val_idx
        )
        val_actual = val_df[meter_id]
        res_pool = build_stratified_residual_pool(val_actual, val_point_pred, block_size=4)
        
        # 2. Test predictions
        test_idx = test_df.index
        rm_test = roll_mean.loc[test_idx].replace(0, np.nan).fillna(s.expanding().mean()).replace(0, 1.0)
        
        feat_test = pd.concat([
            cal_feat.loc[test_idx],
            pd.DataFrame({
                "scaled_lag_96": lag96_cascaded.loc[test_idx] / rm_test,
                "scaled_lag_672": lag672_cascaded.loc[test_idx] / rm_test,
                "scaled_diff": (lag96_cascaded.loc[test_idx] - lag672_cascaded.loc[test_idx]) / rm_test,
                "roll_mean_4": (s.shift(96) / roll_mean).rolling(4).mean().loc[test_idx],
                "roll_mean_96": (s.shift(96) / roll_mean).rolling(96).mean().loc[test_idx],
                "is_fallback": is_fallback.loc[test_idx],
                "macro_grid_ratio": agg_macro_ratio.loc[test_idx],
            }, index=test_idx)
        ], axis=1)
        
        test_point_pred = pd.Series(
            np.clip(model.predict(feat_test) * rm_test.values, 0, None),
            index=test_idx
        )
        test_actual = test_df[meter_id]
        
        # Bootstrap
        sims = generate_stratified_block_bootstrap(test_point_pred, res_pool, n_bootstraps=500, block_size=4)
        quants = extract_prediction_intervals(sims, quantiles=[0.05, 0.10, 0.50, 0.90, 0.95])
        
        # Evaluate 90% interval
        m_eval = evaluate_probabilistic_coverage(test_actual, quants["q_05"], quants["q_95"], nominal_coverage=0.90)
        m_eval["Meter"] = meter_id
        m_eval["Archetype"] = name
        archetype_results[name] = {
            "eval": m_eval,
            "actual": test_actual,
            "pred": test_point_pred,
            "quantiles": quants
        }
        
    df_arch_eval = pd.DataFrame([v["eval"] for v in archetype_results.values()]).set_index("Archetype")
    print("\nArchetype Probabilistic Coverage (90% Nominal Target):")
    print(df_arch_eval.to_string(float_format=lambda x: f"{x:,.2f}"))
    return archetype_results, df_arch_eval


def plot_fan_charts(agg_res: dict, arch_res: dict):
    print("Generating Probabilistic Fan-Chart Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # 1. Aggregate Load Fan-Chart (Summer 1-Week Slice)
    slice_agg = slice("2014-06-09 00:00:00", "2014-06-15 23:45:00")
    t_agg = agg_res["y_test"].loc[slice_agg].index
    y_true_agg = agg_res["y_test"].loc[slice_agg].values
    q05_agg = agg_res["quantiles"]["q_05"][agg_res["y_test"].index.isin(t_agg)]
    q10_agg = agg_res["quantiles"]["q_10"][agg_res["y_test"].index.isin(t_agg)]
    q50_agg = agg_res["quantiles"]["q_50"][agg_res["y_test"].index.isin(t_agg)]
    q90_agg = agg_res["quantiles"]["q_90"][agg_res["y_test"].index.isin(t_agg)]
    q95_agg = agg_res["quantiles"]["q_95"][agg_res["y_test"].index.isin(t_agg)]
    
    axes[0, 0].plot(t_agg, y_true_agg, color="black", linewidth=1.8, label="Actual Grid Load")
    axes[0, 0].plot(t_agg, q50_agg, color="#1e3a8a", linewidth=1.5, linestyle="--", label="Median Forecast (q50)")
    axes[0, 0].fill_between(t_agg, q10_agg, q90_agg, color="#3b82f6", alpha=0.3, label="80% Prediction Band")
    axes[0, 0].fill_between(t_agg, q05_agg, q95_agg, color="#3b82f6", alpha=0.15, label="90% Prediction Band")
    axes[0, 0].set_title("Aggregate Grid Load: Probabilistic Forecast Fan-Chart", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Load (kW)")
    axes[0, 0].legend(loc="upper right", frameon=True)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=25)
    
    # 2. Commercial Office MT_333 Fan-Chart
    c_data = arch_res["Commercial Office"]
    t_c = c_data["actual"].loc[slice_agg].index
    y_true_c = c_data["actual"].loc[slice_agg].values
    idx_mask_c = c_data["actual"].index.isin(t_c)
    
    axes[0, 1].plot(t_c, y_true_c, color="black", linewidth=1.6, label="Actual Load")
    axes[0, 1].plot(t_c, c_data["quantiles"]["q_50"][idx_mask_c], color="#15803d", linewidth=1.5, linestyle="--", label="Median (q50)")
    axes[0, 1].fill_between(t_c, c_data["quantiles"]["q_10"][idx_mask_c], c_data["quantiles"]["q_90"][idx_mask_c], color="#22c55e", alpha=0.3, label="80% Band")
    axes[0, 1].fill_between(t_c, c_data["quantiles"]["q_05"][idx_mask_c], c_data["quantiles"]["q_95"][idx_mask_c], color="#22c55e", alpha=0.15, label="90% Band")
    axes[0, 1].set_title("Commercial Office (MT_333): Probabilistic Fan-Chart", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Load (kW)")
    axes[0, 1].legend(loc="upper right", frameon=True)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis="x", rotation=25)
    
    # 3. Residential Consumer MT_222 Fan-Chart
    r_data = arch_res["Residential Consumer"]
    t_r = r_data["actual"].loc[slice_agg].index
    y_true_r = r_data["actual"].loc[slice_agg].values
    idx_mask_r = r_data["actual"].index.isin(t_r)
    
    axes[1, 0].plot(t_r, y_true_r, color="black", linewidth=1.6, label="Actual Load")
    axes[1, 0].plot(t_r, r_data["quantiles"]["q_50"][idx_mask_r], color="#b45309", linewidth=1.5, linestyle="--", label="Median (q50)")
    axes[1, 0].fill_between(t_r, r_data["quantiles"]["q_10"][idx_mask_r], r_data["quantiles"]["q_90"][idx_mask_r], color="#f59e0b", alpha=0.3, label="80% Band")
    axes[1, 0].fill_between(t_r, r_data["quantiles"]["q_05"][idx_mask_r], r_data["quantiles"]["q_95"][idx_mask_r], color="#f59e0b", alpha=0.15, label="90% Band")
    axes[1, 0].set_title("Residential Consumer (MT_222): Evening Peak Uncertainty Bands", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Load (kW)")
    axes[1, 0].legend(loc="upper right", frameon=True)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis="x", rotation=25)
    
    # 4. Continuous Industrial MT_081 Fan-Chart
    i_data = arch_res["Industrial Plant"]
    t_i = i_data["actual"].loc[slice_agg].index
    y_true_i = i_data["actual"].loc[slice_agg].values
    idx_mask_i = i_data["actual"].index.isin(t_i)
    
    axes[1, 1].plot(t_i, y_true_i, color="black", linewidth=1.6, label="Actual Load")
    axes[1, 1].plot(t_i, i_data["quantiles"]["q_50"][idx_mask_i], color="#7e22ce", linewidth=1.5, linestyle="--", label="Median (q50)")
    axes[1, 1].fill_between(t_i, i_data["quantiles"]["q_10"][idx_mask_i], i_data["quantiles"]["q_90"][idx_mask_i], color="#a855f7", alpha=0.3, label="80% Band")
    axes[1, 1].fill_between(t_i, i_data["quantiles"]["q_05"][idx_mask_i], i_data["quantiles"]["q_95"][idx_mask_i], color="#a855f7", alpha=0.15, label="90% Band")
    axes[1, 1].set_title("Continuous Industrial (MT_081): Tight Baseload Confidence Bounds", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Load (kW)")
    axes[1, 1].legend(loc="upper right", frameon=True)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=25)
    
    plt.tight_layout()
    fan_path = "reports/figures/probabilistic_fan_chart.png"
    plt.savefig(fan_path, bbox_inches="tight")
    plt.close()
    print(f"Saved fan-charts to: {fan_path}")


def main():
    # 1. Aggregate Probabilistic
    agg_res = evaluate_aggregate_probabilistic()
    
    # 2. Archetypes Probabilistic
    df = load_cleaned_data()
    agg = get_aggregate_load(df)
    arch_res, df_arch_eval = evaluate_archetype_probabilistic(df, agg)
    
    # 3. Plot fan-charts
    plot_fan_charts(agg_res, arch_res)
    
    # 4. Save results to CSV
    agg_res["eval_table"].to_csv("data/probabilistic_aggregate_coverage.csv")
    df_arch_eval.to_csv("data/probabilistic_archetypes_coverage.csv")
    print("\nProbabilistic results successfully saved to data/probabilistic_*.csv")


if __name__ == "__main__":
    main()
