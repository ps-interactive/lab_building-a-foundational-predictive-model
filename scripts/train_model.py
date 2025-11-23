#!/usr/bin/env python3
"""
Model Training Script
Trains a logistic regression model for build failure prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import os
import time

def main():
    print("Loading training data...")
    train_df = pd.read_csv('data/train_data.csv')
    
    # Prepare features and target
    X_train = train_df.drop('build_status', axis=1)
    y_train = (train_df['build_status'] == 'success').astype(int)
    
    print(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features")
    print("Target distribution:")
    print(f"  Success: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"  Failure: {(1-y_train).sum()} ({(1-y_train).sum()/len(y_train)*100:.1f}%)")
    
    print("Initializing Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    print("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print("Model training complete!")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/logistic_regression_model.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Also save feature names for later use
    feature_names_path = 'models/feature_names.txt'
    with open(feature_names_path, 'w') as f:
        for feature in X_train.columns:
            f.write(f"{feature}\n")
    
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
