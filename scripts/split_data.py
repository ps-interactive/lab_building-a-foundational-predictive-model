#!/usr/bin/env python3
"""
Data Splitting Script
Splits data into training and testing sets with stratification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def main():
    print("Loading feature-engineered data...")
    df = pd.read_csv('data/pipeline_features.csv')
    print(f"Total samples: {df.shape[0]}")
    
    # Prepare features and target
    X = df.drop(['build_id', 'build_status'], axis=1)
    y = df['build_status']
    
    # Convert target to binary (0 = failure, 1 = success)
    y_binary = (y == 'success').astype(int)
    
    print("Splitting data with 80/20 train/test ratio...")
    print("Using stratification to maintain class balance...")
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Add back build_id and original target
    train_df = X_train.copy()
    train_df['build_status'] = y_train.map({0: 'failure', 1: 'success'})
    
    test_df = X_test.copy()
    test_df['build_status'] = y_test.map({0: 'failure', 1: 'success'})
    
    print(f"Training set size: {len(train_df)} samples")
    print(f"Testing set size: {len(test_df)} samples")
    
    # Show class distribution
    print("Training set class distribution:")
    train_counts = train_df['build_status'].value_counts()
    print(f"  Success: {train_counts['success']} ({train_counts['success']/len(train_df)*100:.1f}%)")
    print(f"  Failure: {train_counts['failure']} ({train_counts['failure']/len(train_df)*100:.1f}%)")
    
    print("Testing set class distribution:")
    test_counts = test_df['build_status'].value_counts()
    print(f"  Success: {test_counts['success']} ({test_counts['success']/len(test_df)*100:.1f}%)")
    print(f"  Failure: {test_counts['failure']} ({test_counts['failure']/len(test_df)*100:.1f}%)")
    
    # Save datasets
    os.makedirs('data', exist_ok=True)
    train_path = 'data/train_data.csv'
    test_path = 'data/test_data.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("Data split complete!")
    print(f"Training data saved to: {train_path}")
    print(f"Testing data saved to: {test_path}")

if __name__ == "__main__":
    main()
