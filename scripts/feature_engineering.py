#!/usr/bin/env python3
"""
Feature Engineering Script
Creates rolling average features from time-series data
"""

import pandas as pd
import numpy as np
import os

def main():
    print("Loading cleaned data...")
    df = pd.read_csv('data/pipeline_logs_cleaned.csv')
    print(f"Dataset shape: {df.shape}")
    
    print("Calculating rolling averages...")
    
    # Sort by build_id to ensure chronological order
    df = df.sort_values('build_id').reset_index(drop=True)
    
    # Create rolling averages with window of 5
    window_size = 5
    
    print(f"Creating rolling_avg_test_duration (window={window_size})")
    df['rolling_avg_test_duration'] = df['test_duration'].rolling(window=window_size, min_periods=1).mean()
    
    print(f"Creating rolling_avg_cpu_usage (window={window_size})")
    df['rolling_avg_cpu_usage'] = df['cpu_usage'].rolling(window=window_size, min_periods=1).mean()
    
    print(f"Creating rolling_avg_memory_usage (window={window_size})")
    df['rolling_avg_memory_usage'] = df['memory_usage'].rolling(window=window_size, min_periods=1).mean()
    
    # Remove rows where rolling averages couldn't be properly calculated
    df = df.dropna()
    
    print("Feature engineering complete!")
    print(f"New dataset shape: {df.shape}")
    
    # Save feature-engineered data
    os.makedirs('data', exist_ok=True)
    output_path = 'data/pipeline_features.csv'
    df.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")

if __name__ == "__main__":
    main()
