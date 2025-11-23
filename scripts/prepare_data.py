#!/usr/bin/env python3
"""
Data Preparation Script
Cleans pipeline logs data by handling missing values and removing outliers
"""

import pandas as pd
import numpy as np
import os

def main():
    print("Loading pipeline logs data...")
    df = pd.read_csv('data/pipeline_logs.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Check for missing values
    print("Checking for missing values...")
    missing_count = df.isnull().sum().sum()
    print(f"Missing values found: {missing_count}")
    
    if missing_count > 0:
        print("Handling missing values...")
        # Fill numerical columns with median
        numerical_cols = ['test_duration', 'dependency_count', 'cpu_usage', 'memory_usage']
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    # Remove outliers using IQR method
    print("Checking for outliers in numerical columns...")
    initial_shape = df.shape[0]
    
    for col in ['test_duration', 'cpu_usage', 'memory_usage']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    outliers_removed = initial_shape - df.shape[0]
    print(f"Outliers removed: {outliers_removed}")
    print(f"Final dataset shape: {df.shape}")
    
    # Save cleaned data
    os.makedirs('data', exist_ok=True)
    output_path = 'data/pipeline_logs_cleaned.csv'
    df.to_csv(output_path, index=False)
    print("Data preparation complete!")
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    main()
