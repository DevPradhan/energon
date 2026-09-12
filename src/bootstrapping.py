"""Empirical Stratified Residual Bootstrapping and Probabilistic Forecasting Module.

Provides:
- Stratified residual pool construction by (hour_of_day, is_weekend)
- Contiguous block bootstrapping to preserve residual autocorrelation
- Calibrated prediction intervals (80%, 90%, 95%) and peak capacity reserves
- Probabilistic evaluation metrics: Empirical Coverage (PICP), Width (MPIW), and Winkler Score
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def build_stratified_residual_pool(
    y_true: pd.Series,
    y_pred: pd.Series,
    block_size: int = 4
) -> Dict[Tuple[int, int], List[np.ndarray]]:
    """Construct stratified contiguous residual blocks by (hour, is_weekend).
    
    Parameters
    ----------
    y_true : pd.Series
        Ground truth load during validation period.
    y_pred : pd.Series
        Model prediction during validation period.
    block_size : int
        Length of contiguous blocks (e.g. 4 intervals = 1 hour) to retain autocorrelation.
        
    Returns
    -------
    pool : Dict[(hour, is_weekend), List[np.ndarray]]
        Lookup table of contiguous residual blocks for each diurnal stratum.
    """
    valid_mask = np.isfinite(y_true.values) & np.isfinite(y_pred.values)
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    
    residuals = yt - yp
    hours = yt.index.hour
    is_weekends = yt.index.dayofweek.isin([5, 6]).astype(int)
    
    pool = {(h, w): [] for h in range(24) for w in [0, 1]}
    
    # Extract contiguous blocks
    n_samples = len(residuals)
    for i in range(0, n_samples - block_size + 1):
        # Check if timestamps are contiguous (15 min apart)
        time_diff = (residuals.index[i + block_size - 1] - residuals.index[i]).total_seconds()
        expected_diff = (block_size - 1) * 15 * 60
        
        if time_diff == expected_diff:
            start_h = hours[i]
            start_w = is_weekends[i]
            block = residuals.iloc[i : i + block_size].values
            pool[(start_h, start_w)].append(block)
            
    # Fallback: if any stratum has fewer than 10 blocks, backfill from adjacent hours
    for (h, w), blocks in pool.items():
        if len(blocks) < 10:
            for adj_h in [(h - 1) % 24, (h + 1) % 24]:
                pool[(h, w)].extend(pool.get((adj_h, w), []))
                
    return pool


def generate_stratified_block_bootstrap(
    point_preds: pd.Series,
    residual_pool: Dict[Tuple[int, int], List[np.ndarray]],
    n_bootstraps: int = 500,
    block_size: int = 4,
    random_seed: int = 42
) -> np.ndarray:
    """Generate simulated trajectory paths using stratified block bootstrapping.
    
    Parameters
    ----------
    point_preds : pd.Series
        Base point forecast (e.g. from LightGBM).
    residual_pool : dict
        Stratified pool of historical residual blocks.
    n_bootstraps : int
        Number of bootstrap trajectory draws.
    block_size : int
        Block size (in 15-min intervals).
        
    Returns
    -------
    simulations : np.ndarray of shape (len(point_preds), n_bootstraps)
        Simulated non-negative power load realizations.
    """
    rng = np.random.default_rng(random_seed)
    n_steps = len(point_preds)
    timestamps = point_preds.index
    
    hours = timestamps.hour
    is_weekends = timestamps.dayofweek.isin([5, 6]).astype(int)
    
    simulated_residuals = np.zeros((n_steps, n_bootstraps), dtype=np.float32)
    
    # Step through series in blocks
    idx = 0
    while idx < n_steps:
        cur_h = hours[idx]
        cur_w = is_weekends[idx]
        cur_len = min(block_size, n_steps - idx)
        
        available_blocks = residual_pool.get((cur_h, cur_w), [])
        if len(available_blocks) == 0:
            # Universal fallback
            all_blocks = [b for b_list in residual_pool.values() for b in b_list]
            sampled_idx = rng.integers(0, len(all_blocks), size=n_bootstraps)
            sampled_blocks = np.array([all_blocks[k][:cur_len] for k in sampled_idx]) # (B, cur_len)
        else:
            sampled_idx = rng.integers(0, len(available_blocks), size=n_bootstraps)
            sampled_blocks = np.array([available_blocks[k][:cur_len] for k in sampled_idx]) # (B, cur_len)
            
        simulated_residuals[idx : idx + cur_len, :] = sampled_blocks.T
        idx += cur_len
        
    # Add residuals to base point forecast and enforce non-negativity
    base_values = point_preds.values[:, np.newaxis]
    simulations = np.clip(base_values + simulated_residuals, 0.0, None)
    
    return simulations


def extract_prediction_intervals(
    simulations: np.ndarray,
    quantiles: List[float] = [0.05, 0.10, 0.50, 0.90, 0.95]
) -> Dict[str, np.ndarray]:
    """Compute quantile trajectories from bootstrap simulation ensemble."""
    results = {}
    for q in quantiles:
        name = f"q_{int(q * 100):02d}"
        results[name] = np.quantile(simulations, q, axis=1)
    return results


def evaluate_probabilistic_coverage(
    y_true: pd.Series,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    nominal_coverage: float = 0.90
) -> Dict[str, float]:
    """Compute empirical coverage, interval sharpness, and Winkler score."""
    yt = np.asarray(y_true.values, dtype=np.float64)
    lb = np.asarray(lower_bound, dtype=np.float64)
    ub = np.asarray(upper_bound, dtype=np.float64)
    
    mask = np.isfinite(yt) & np.isfinite(lb) & np.isfinite(ub)
    yt, lb, ub = yt[mask], lb[mask], ub[mask]
    
    if len(yt) == 0:
        return {"PICP (%)": np.nan, "MPIW (kW)": np.nan, "Winkler Score": np.nan}
        
    # 1. Prediction Interval Coverage Probability (PICP)
    covered = (yt >= lb) & (yt <= ub)
    picp = float(np.mean(covered) * 100.0)
    
    # 2. Mean Prediction Interval Width (MPIW)
    widths = ub - lb
    mpiw = float(np.mean(widths))
    mean_true = float(np.mean(yt))
    nmpiw = float((mpiw / mean_true * 100.0) if mean_true > 0 else np.nan)
    
    # 3. Winkler Interval Score
    # Penalizes both width and out-of-bounds misses proportionally
    alpha = 1.0 - nominal_coverage
    penalty_lower = (2.0 / alpha) * (lb - yt) * (yt < lb)
    penalty_upper = (2.0 / alpha) * (yt - ub) * (yt > ub)
    winkler = float(np.mean(widths + penalty_lower + penalty_upper))
    
    return {
        "Nominal Target (%)": nominal_coverage * 100.0,
        "Empirical PICP (%)": picp,
        "Coverage Gap (%)": picp - (nominal_coverage * 100.0),
        "MPIW (kW)": mpiw,
        "Normalized MPIW (%)": nmpiw,
        "Winkler Score": winkler,
    }
