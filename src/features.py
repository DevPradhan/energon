"""Feature engineering module for electricity load forecasting.

Generates temporal/cyclical features, horizon-constrained autoregressive lags,
and rolling statistics for Day-Ahead (H=96) and Week-Ahead (H=672) forecasting.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def create_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate calendar and cyclical trigonometric features from DatetimeIndex."""
    df = pd.DataFrame(index=index)
    
    # Minute of day (0 to 1439)
    minute_of_day = index.hour * 60 + index.minute
    df["sin_time_of_day"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["cos_time_of_day"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    
    # Day of week (0=Monday, 6=Sunday)
    day_of_week = index.dayofweek
    df["sin_day_of_week"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["cos_day_of_week"] = np.cos(2 * np.pi * day_of_week / 7.0)
    df["day_of_week"] = day_of_week
    df["is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    
    # Day of year (annual seasonality)
    day_of_year = index.dayofyear
    df["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["month"] = index.month
    
    return df


def create_aggregate_features(
    series: pd.Series,
    horizon: str = "day_ahead"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate complete feature matrix X and target y for aggregate load forecasting.
    
    Parameters
    ----------
    series : pd.Series
        Cleaned aggregate electricity load series.
    horizon : str
        'day_ahead' (H=96 intervals / 24 hours) or 'week_ahead' (H=672 intervals / 7 days).
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix aligned with series index.
    y : pd.Series
        Target series.
    """
    cal_features = create_calendar_features(series.index)
    lag_df = pd.DataFrame(index=series.index)
    
    if horizon == "day_ahead":
        # Day-Ahead: Minimum valid lag is 96 (24 hours prior)
        # 1. Primary diurnal & seasonal lags
        lag_df["lag_96"] = series.shift(96)     # 24 hours ago
        lag_df["lag_192"] = series.shift(192)   # 48 hours ago
        lag_df["lag_288"] = series.shift(288)   # 72 hours ago
        lag_df["lag_672"] = series.shift(672)   # 7 days ago (same day of week)
        lag_df["lag_1344"] = series.shift(1344) # 14 days ago
        
        # 2. Inter-day differences / trend
        lag_df["diff_96_192"] = lag_df["lag_96"] - lag_df["lag_192"]
        lag_df["diff_96_672"] = lag_df["lag_96"] - lag_df["lag_672"]
        
        # 3. Rolling window statistics on past data (shifted by 96)
        shifted_96 = series.shift(96)
        lag_df["roll_mean_4"] = shifted_96.rolling(4).mean()   # 1-hour window at t-96
        lag_df["roll_mean_96"] = shifted_96.rolling(96).mean() # 24-hour window at t-96
        lag_df["roll_std_96"] = shifted_96.rolling(96).std()
        
        # 4. 4-week seasonal same-slot mean (lags 7d, 14d, 21d, 28d)
        lag_df["seasonal_4w_mean"] = pd.concat([
            series.shift(672),
            series.shift(1344),
            series.shift(2016),
            series.shift(2688)
        ], axis=1).mean(axis=1)

    elif horizon == "week_ahead":
        # Week-Ahead: Minimum valid lag is 672 (7 days prior)
        lag_df["lag_672"] = series.shift(672)   # 7 days ago
        lag_df["lag_1344"] = series.shift(1344) # 14 days ago
        lag_df["lag_2016"] = series.shift(2016) # 21 days ago
        lag_df["lag_2688"] = series.shift(2688) # 28 days ago
        
        # Differences
        lag_df["diff_672_1344"] = lag_df["lag_672"] - lag_df["lag_1344"]
        
        # Rolling window stats on shifted series
        shifted_672 = series.shift(672)
        lag_df["roll_mean_96"] = shifted_672.rolling(96).mean()
        lag_df["roll_mean_672"] = shifted_672.rolling(672).mean()
        lag_df["roll_std_672"] = shifted_672.rolling(672).std()
        
        # 4-week seasonal mean
        lag_df["seasonal_4w_mean"] = pd.concat([
            series.shift(672),
            series.shift(1344),
            series.shift(2016),
            series.shift(2688)
        ], axis=1).mean(axis=1)
    else:
        raise ValueError(f"Unknown horizon: {horizon}. Must be 'day_ahead' or 'week_ahead'")

    X = pd.concat([cal_features, lag_df], axis=1)
    y = series.copy()
    
    return X, y
