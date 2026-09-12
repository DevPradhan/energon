import pandas as pd
import numpy as np
import time
import os
import subprocess

print("Loading dataset...")
start = time.time()
df = pd.read_csv('data/electricityloaddiagrams20112014/LD2011_2014.txt', sep=';', decimal=',', parse_dates=[0], index_col=0, low_memory=False)
print(f"Loaded in {time.time() - start:.2f}s")

# Ensure float32 to save memory
df = df.astype(np.float32)

print("Handling pre-online zeros...")
df_replaced = df.replace(0, np.nan)
cleaned_data = {}

for col in df.columns:
    s = df[col].copy()
    
    # 1. Pre-online to NaN
    first_idx = df_replaced[col].first_valid_index()
    if first_idx is None:
        cleaned_data[col] = s
        continue
    
    # Set pre-online to NaN
    s.loc[:first_idx] = s.loc[:first_idx].replace(0, np.nan)
    
    # 2. Identify contiguous blocks of 0s post-online
    is_zero = s == 0
    blocks = (is_zero != is_zero.shift()).cumsum()
    
    # Filter only zero blocks
    zero_blocks = blocks[is_zero]
    
    # Get size of each block
    block_sizes = zero_blocks.map(zero_blocks.value_counts())
    
    # Set all post-online 0s to NaN so we can fill them appropriately
    s[is_zero] = np.nan
    
    # Block sizes in intervals (15 mins each)
    short_blips = is_zero & (block_sizes <= 4)
    medium_blips = is_zero & (block_sizes > 4) & (block_sizes <= 96)
    
    # For short blips: global linear interpolation, keep only for short blips
    s_linear = s.interpolate(method='linear')
    s.loc[short_blips] = s_linear.loc[short_blips]
    
    # For medium blips: seasonal imputation (previous 24 hours / 96 intervals)
    s_seasonal = s.shift(96)
    s.loc[medium_blips] = s_seasonal.loc[medium_blips]
    
    # Long blips (> 96) automatically remain NaN because we set is_zero to NaN above
    
    cleaned_data[col] = s

df_clean = pd.DataFrame(cleaned_data)

# Print some verification stats
print("\n--- Verification ---")
print(f"Original missing/NaN count: {df.isna().sum().sum()}")
print(f"New missing/NaN count: {df_clean.isna().sum().sum()}")
print(f"Original total 0s: {(df == 0).sum().sum()}")
print(f"New total 0s: {(df_clean == 0).sum().sum()}")

print("\nSaving to parquet...")
output_path = 'data/cleaned_electricity_load.parquet'
try:
    df_clean.to_parquet(output_path, engine='pyarrow')
except ImportError:
    print("Installing pyarrow...")
    subprocess.check_call(["pip", "install", "pyarrow"])
    df_clean.to_parquet(output_path, engine='pyarrow')

print(f"Done! Cleaned dataset saved to {output_path}")
