#!/usr/bin/env python3
"""
Test Script for Scikit-learn Functionality
Validates that scikit-learn is working correctly with basic operations
"""

import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def main():
    print("Testing scikit-learn basic functionality...")
    
    try:
        # Generate a simple dataset
        print("Generating test dataset...")
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train a simple model
        print("Training logistic regression model...")
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        print("Model trained successfully!")
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(X[:5])
        print(f"Sample predictions: {predictions}")
        
        # Calculate accuracy
        score = model.score(X, y)
        print(f"Training accuracy: {score:.2f}")
        
        print("\n✓ All tests passed! Scikit-learn is working correctly.")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
