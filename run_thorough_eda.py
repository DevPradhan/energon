"""Comprehensive Exploratory Data Analysis (EDA) for Electricity Load Forecasting.

Generates:
1. Aggregate Load Timeline & Seasonality Plots
2. Diurnal / Weekly / Monthly Profiles
3. Meter Heterogeneity & Scale Analysis
4. Consumer Archetypes (Clustering / Load Signatures)
5. Temporal Correlation (ACF / PACF)
6. Statistical Summary Report
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf, adfuller

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    print("Loading data from parquet...")
    df = pd.read_parquet("data/cleaned_electricity_load.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    agg = df.sum(axis=1, min_count=1)
    agg.name = "aggregate_load"
    return df, agg


def analyze_aggregate_timeline(df: pd.DataFrame, agg: pd.Series):
    print("Plotting 1: Aggregate Timeline & Active Meters...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    # Daily resampled mean aggregate load
    agg_daily = agg.resample("D").mean()
    agg_daily_roll7 = agg_daily.rolling(7, center=True).mean()
    
    ax1.plot(agg_daily.index, agg_daily.values, color="#94a3b8", alpha=0.5, label="Daily Mean Load (kW)")
    ax1.plot(agg_daily_roll7.index, agg_daily_roll7.values, color="#2563eb", linewidth=2, label="7-Day Smoothed Load (kW)")
    ax1.axvline(pd.to_datetime("2012-01-01"), color="#dc2626", linestyle="--", alpha=0.7, label="Stable Baseline (2012-01-01)")
    ax1.axvline(pd.to_datetime("2014-01-01"), color="#16a34a", linestyle="--", alpha=0.7, label="Test Set Split (2014-01-01)")
    ax1.set_title("System-Wide Aggregate Electricity Load (2011–2014)", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Average Load (kW)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Active meter count over time
    active_meters = df.notna().sum(axis=1).resample("D").median()
    ax2.plot(active_meters.index, active_meters.values, color="#d97706", linewidth=1.8)
    ax2.set_title("Active Meters Online Over Time (Total = 370)", fontsize=11, fontweight="bold", pad=6)
    ax2.set_ylabel("Meters Active", fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(0, 390)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "eda_01_aggregate_load_timeline.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def analyze_seasonal_profiles(agg: pd.Series):
    print("Plotting 2: Seasonal Profiles (Diurnal, Weekly, Monthly)...")
    # Restrict to stable 2012-2014 period
    agg_stable = agg.loc["2012-01-01":"2014-12-31"]
    
    df_feat = pd.DataFrame({"load": agg_stable})
    df_feat["hour"] = df_feat.index.hour + df_feat.index.minute / 60.0
    df_feat["day_of_week"] = df_feat.index.day_name()
    df_feat["day_idx"] = df_feat.index.dayofweek
    df_feat["is_weekend"] = df_feat["day_idx"].isin([5, 6])
    df_feat["month"] = df_feat.index.month_name()
    df_feat["month_idx"] = df_feat.index.month
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Panel 1: Diurnal 24-hour Profile (Weekday vs Weekend)
    weekday_diurnal = df_feat[~df_feat["is_weekend"]].groupby("hour")["load"]
    weekend_diurnal = df_feat[df_feat["is_weekend"]].groupby("hour")["load"]
    
    axes[0].plot(weekday_diurnal.mean().index, weekday_diurnal.mean().values, color="#2563eb", linewidth=2.5, label="Weekday Mean")
    axes[0].fill_between(weekday_diurnal.mean().index, 
                         weekday_diurnal.quantile(0.10).values, 
                         weekday_diurnal.quantile(0.90).values, 
                         color="#2563eb", alpha=0.15, label="Weekday 10-90th Pct")
    
    axes[0].plot(weekend_diurnal.mean().index, weekend_diurnal.mean().values, color="#ea580c", linewidth=2.5, label="Weekend Mean")
    axes[0].fill_between(weekend_diurnal.mean().index, 
                         weekend_diurnal.quantile(0.10).values, 
                         weekend_diurnal.quantile(0.90).values, 
                         color="#ea580c", alpha=0.15, label="Weekend 10-90th Pct")
    
    axes[0].set_title("24-Hour Diurnal Demand Profile", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hour of Day", fontsize=10)
    axes[0].set_ylabel("Aggregate Load (kW)", fontsize=10)
    axes[0].set_xticks(range(0, 25, 4))
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Day of Week Boxplot
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.boxplot(data=df_feat, x="day_of_week", y="load", order=day_order, ax=axes[1], palette="Blues_r", showfliers=False)
    axes[1].set_title("Load by Day of Week", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Load (kW)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Monthly Seasonality Boxplot
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    sns.boxplot(data=df_feat, x="month", y="load", order=month_order, ax=axes[2], palette="Spectral_r", showfliers=False)
    axes[2].set_title("Annual Seasonality (Monthly Distribution)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Load (kW)", fontsize=10)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "eda_02_seasonal_profiles.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def analyze_meter_heterogeneity(df: pd.DataFrame, agg: pd.Series):
    print("Plotting 3: Individual Meter Heterogeneity...")
    # Compute per-meter statistics on stable 2012-2014 period
    df_stable = df.loc["2012-01-01":"2014-12-31"]
    agg_stable = agg.loc["2012-01-01":"2014-12-31"]
    
    mean_load = df_stable.mean()
    max_load = df_stable.max()
    load_factor = mean_load / max_load  # ratio of average to peak load (measure of load stability)
    corrs = df_stable.corrwith(agg_stable)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Distribution of Mean Load (Log Scale)
    sns.histplot(mean_load, bins=40, kde=True, ax=axes[0], color="#0284c7", log_scale=True)
    axes[0].set_title("Consumer Scale Distribution (Mean kW)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean Load (kW) [Log Scale]", fontsize=10)
    axes[0].set_ylabel("Number of Meters", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Load Factor (Stability: Mean / Peak)
    sns.histplot(load_factor, bins=35, kde=True, ax=axes[1], color="#059669")
    axes[1].set_title("Load Factor Distribution (Mean / Peak)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Load Factor (Higher = Flatter 24/7 Demand)", fontsize=10)
    axes[1].set_ylabel("Number of Meters", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Correlation with Aggregate Load
    sns.histplot(corrs, bins=35, kde=True, ax=axes[2], color="#7c3aed")
    axes[2].axvline(corrs.median(), color="#dc2626", linestyle="--", label=f"Median Corr = {corrs.median():.2f}")
    axes[2].set_title("Meter Correlation with Aggregate Grid Load", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Pearson Correlation (r)", fontsize=10)
    axes[2].set_ylabel("Number of Meters", fontsize=10)
    axes[2].legend(loc="upper left", frameon=True)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "eda_03_meter_heterogeneity.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def analyze_consumer_archetypes(df: pd.DataFrame):
    print("Plotting 4: Consumer Archetypes & Load Signatures...")
    # Let's identify 4 distinct archetypes from the dataset based on diurnal & weekly features:
    # 1. Commercial / Office (strong weekday 9-5 peak, massive weekend drop)
    # 2. Residential (evening peak, lower daytime, steady weekends)
    # 3. Industrial / Flat Baselines (high load factor, minimal weekend drop)
    # 4. Night-shift / Off-peak heavy
    
    df_stable = df.loc["2013-01-01":"2013-12-31"]
    is_weekend = df_stable.index.dayofweek.isin([5, 6])
    hours = df_stable.index.hour
    
    weekday_means = df_stable[~is_weekend].mean()
    weekend_means = df_stable[is_weekend].mean()
    weekend_ratio = weekend_means / weekday_means.replace(0, np.nan)
    
    day_means = df_stable[(hours >= 9) & (hours <= 17)].mean()
    night_means = df_stable[(hours >= 19) & (hours <= 23)].mean()
    evening_to_day_ratio = night_means / day_means.replace(0, np.nan)
    
    # Select archetypes:
    commercial_meter = weekend_ratio.sort_values().index[0]  # biggest weekend drop
    residential_meter = evening_to_day_ratio.sort_values(ascending=False).index[5] # strong evening peak
    
    load_factor = df_stable.mean() / df_stable.max()
    industrial_meter = load_factor[weekend_ratio > 0.85].sort_values(ascending=False).index[0] # steady 24/7
    
    # Pick a high-volatility intermittent meter
    cv = df_stable.std() / df_stable.mean()
    volatile_meter = cv.sort_values(ascending=False).index[5]
    
    archetypes = [
        ("Commercial / Office Building", commercial_meter, "#2563eb"),
        ("Residential Consumer", residential_meter, "#ea580c"),
        ("Continuous 24/7 Industrial Load", industrial_meter, "#059669"),
        ("Peaky / Intermittent Consumer", volatile_meter, "#9333ea")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    
    for idx, (label, meter_id, color) in enumerate(archetypes):
        ax = axes[idx]
        m_series = df_stable[meter_id]
        m_df = pd.DataFrame({"load": m_series})
        m_df["hour"] = m_df.index.hour + m_df.index.minute / 60.0
        m_df["is_weekend"] = m_df.index.dayofweek.isin([5, 6])
        
        wk_diurnal = m_df[~m_df["is_weekend"]].groupby("hour")["load"].mean()
        we_diurnal = m_df[m_df["is_weekend"]].groupby("hour")["load"].mean()
        
        ax.plot(wk_diurnal.index, wk_diurnal.values, color=color, linewidth=2.5, label="Weekday Mean")
        ax.plot(we_diurnal.index, we_diurnal.values, color="#64748b", linestyle="--", linewidth=2, label="Weekend Mean")
        ax.set_title(f"Archetype {idx+1}: {label} ({meter_id})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Power Demand (kW)", fontsize=10)
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("Hour of Day", fontsize=10)
            ax.set_xticks(range(0, 25, 4))
            
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "eda_04_consumer_archetypes.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def analyze_autocorrelation(agg: pd.Series):
    print("Plotting 5: ACF and PACF Analysis...")
    agg_stable = agg.loc["2012-01-01":"2014-12-31"]
    
    # 2 weeks of lags at 15-min frequency = 14 * 96 = 1344
    n_lags = 1344
    acf_vals = acf(agg_stable.dropna(), nlags=n_lags, fft=True)
    pacf_vals = pacf(agg_stable.dropna(), nlags=96, method="ywm")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # ACF Plot
    lags_hours = np.arange(len(acf_vals)) * 15.0 / 60.0
    ax1.plot(lags_hours, acf_vals, color="#2563eb", linewidth=1.5)
    
    # Annotate daily and weekly cycles
    for day in range(1, 15):
        h = day * 24
        if day in [1, 7, 14]:
            ax1.axvline(h, color="#dc2626" if day in [7, 14] else "#ea580c", linestyle="--", alpha=0.7)
            ax1.text(h + 1, 0.85 if day == 1 else 0.75, f"{day}d lag\n(r={acf_vals[day * 96]:.2f})", 
                     fontsize=9, fontweight="bold", color="#dc2626" if day in [7, 14] else "#ea580c")
        else:
            ax1.axvline(h, color="#cbd5e1", linestyle=":", alpha=0.5)
            
    ax1.set_title("Autocorrelation Function (ACF) up to 14 Days (1,344 Lags @ 15-Min)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Autocorrelation (r)", fontsize=10)
    ax1.set_xlabel("Lag (Hours)", fontsize=10)
    ax1.set_xlim(0, 14 * 24)
    ax1.grid(True, alpha=0.3)
    
    # PACF Plot up to 24 hours (96 lags)
    pacf_lags = np.arange(len(pacf_vals)) * 15.0 / 60.0
    ax2.bar(pacf_lags, pacf_vals, width=0.2, color="#059669", edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axvline(24, color="#dc2626", linestyle="--", alpha=0.7, label="24h Lag (96 intervals)")
    ax2.set_title("Partial Autocorrelation Function (PACF) up to 24 Hours", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Partial Autocorrelation", fontsize=10)
    ax2.set_xlabel("Lag (Hours)", fontsize=10)
    ax2.set_xlim(0, 24.5)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "eda_05_autocorrelation_acf_pacf.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def compute_summary_stats(df: pd.DataFrame, agg: pd.Series):
    print("Computing statistical summary and stationarity tests...")
    agg_stable = agg.loc["2012-01-01":"2014-12-31"]
    
    # ADF test on raw series and seasonal difference
    adf_raw = adfuller(agg_stable.resample("h").mean().dropna())
    # Seasonal difference (24-hour diff on hourly)
    agg_hourly = agg_stable.resample("h").mean().dropna()
    adf_diff = adfuller((agg_hourly - agg_hourly.shift(24)).dropna())
    
    df_stable = df.loc["2012-01-01":"2014-12-31"]
    means = df_stable.mean()
    maxs = df_stable.max()
    load_factors = (means / maxs.replace(0, np.nan)).dropna()
    corrs = df_stable.corrwith(agg_stable).dropna()
    
    summary = {
        "dataset": {
            "total_meters": len(df.columns),
            "time_range": [str(df.index.min()), str(df.index.max())],
            "total_intervals": len(df),
            "stable_period_start": "2012-01-01",
        },
        "aggregate_load": {
            "mean_kw": float(agg_stable.mean()),
            "median_kw": float(agg_stable.median()),
            "std_kw": float(agg_stable.std()),
            "min_kw": float(agg_stable.min()),
            "max_kw": float(agg_stable.max()),
            "peak_to_average_ratio": float(agg_stable.max() / agg_stable.mean()),
            "weekday_mean_kw": float(agg_stable[~agg_stable.index.dayofweek.isin([5, 6])].mean()),
            "weekend_mean_kw": float(agg_stable[agg_stable.index.dayofweek.isin([5, 6])].mean()),
            "weekend_drop_pct": float((1 - agg_stable[agg_stable.index.dayofweek.isin([5, 6])].mean() / 
                                       agg_stable[~agg_stable.index.dayofweek.isin([5, 6])].mean()) * 100),
            "adf_raw_pvalue": float(adf_raw[1]),
            "adf_seasonal_diff_pvalue": float(adf_diff[1]),
        },
        "meter_heterogeneity": {
            "min_meter_mean_kw": float(means.min()),
            "median_meter_mean_kw": float(means.median()),
            "mean_meter_mean_kw": float(means.mean()),
            "max_meter_mean_kw": float(means.max()),
            "dynamic_scale_ratio": float(means.max() / means.min()),
            "mean_load_factor": float(load_factors.mean()),
            "median_load_factor": float(load_factors.median()),
            "median_corr_with_agg": float(corrs.median()),
            "p25_corr_with_agg": float(corrs.quantile(0.25)),
            "p75_corr_with_agg": float(corrs.quantile(0.75)),
        }
    }
    
    with open("reports/eda_summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Saved: reports/eda_summary_statistics.json")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    df, agg = load_data()
    analyze_aggregate_timeline(df, agg)
    analyze_seasonal_profiles(agg)
    analyze_meter_heterogeneity(df, agg)
    analyze_consumer_archetypes(df)
    analyze_autocorrelation(agg)
    compute_summary_stats(df, agg)
    print("\nEDA Completed successfully!")


if __name__ == "__main__":
    main()
