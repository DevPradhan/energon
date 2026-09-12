"""Evaluation metrics for time-series electricity forecasting."""

import numpy as np
import pandas as pd


def _filter_valid(y_true, y_pred):
    """Filter out NaNs from paired true and predicted values."""
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def wmape(y_true, y_pred) -> float:
    """Weighted Mean Absolute Percentage Error (WMAPE / Normalized MAE).
    
    Formula: sum(|y_true - y_pred|) / sum(|y_true|)
    """
    yt, yp = _filter_valid(y_true, y_pred)
    if len(yt) == 0:
        return np.nan
    sum_true = np.sum(np.abs(yt))
    if sum_true == 0:
        return np.nan
    return float(np.sum(np.abs(yt - yp)) / sum_true)


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    yt, yp = _filter_valid(y_true, y_pred)
    if len(yt) == 0:
        return np.nan
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    yt, yp = _filter_valid(y_true, y_pred)
    if len(yt) == 0:
        return np.nan
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def nrmse(y_true, y_pred) -> float:
    """Normalized Root Mean Squared Error (normalized by mean of y_true)."""
    yt, yp = _filter_valid(y_true, y_pred)
    if len(yt) == 0:
        return np.nan
    mean_val = np.mean(yt)
    if mean_val == 0:
        return np.nan
    return float(np.sqrt(np.mean((yt - yp) ** 2)) / mean_val)


def evaluate_all(y_true, y_pred) -> dict:
    """Compute all standard forecasting metrics."""
    return {
        "WMAPE": wmape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
    }
