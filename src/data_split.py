"""Data splitting and preparation utilities for electricity forecasting."""

import pandas as pd
from typing import Tuple, Dict, Optional


def load_cleaned_data(filepath: str = "data/cleaned_electricity_load.parquet") -> pd.DataFrame:
    """Load the cleaned 15-minute electricity load parquet dataset."""
    df = pd.read_parquet(filepath)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def get_aggregate_load(df: pd.DataFrame) -> pd.Series:
    """Compute aggregate grid load by summing all active meters at each timestamp.
    
    If all meters are NaN at a timestamp, returns NaN. Otherwise sums valid meters.
    """
    agg = df.sum(axis=1, min_count=1)
    agg.name = "aggregate_load"
    return agg


def split_data(
    data: pd.DataFrame | pd.Series,
    train_start: str = "2012-01-01 00:15:00",
    val_start: str = "2013-10-01 00:00:00",
    test_start: str = "2014-01-01 00:00:00",
    test_end: str = "2014-12-31 23:45:00",
) -> Dict[str, pd.DataFrame | pd.Series]:
    """Split dataset chronologically into Train, Validation, and Test sets.
    
    Default ranges:
    - Train: 2012-01-01 00:15:00 to 2013-09-30 23:45:00
    - Val:   2013-10-01 00:00:00 to 2013-12-31 23:45:00 (Q4 2013)
    - Test:  2014-01-01 00:00:00 to 2014-12-31 23:45:00 (Full year 2014, exactly 35,040 steps)
    """
    train = data.loc[train_start:val_start].iloc[:-1]
    val = data.loc[val_start:test_start].iloc[:-1]
    test = data.loc[test_start:test_end]
    
    return {
        "train": train,
        "val": val,
        "test": test,
    }
