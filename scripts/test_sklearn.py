#!/usr/bin/env python3
"""
Scikit-learn Basic Functionality Test
Tests that scikit-learn is working correctly with basic ML operations
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("="*50)
    print("Testing Scikit-learn Basic Functionality")
    print("="*50 + "\n")
    
    # Create a simple synthetic dataset
    print("1. Creating synthetic dataset...")
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   ✓ Dataset created successfully\n")
    
    # Split the data
    print("2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    print(f"   ✓ Data split successful\n")
    
    # Train a model
    print("3. Training logistic regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    print(f"   Model coefficients shape: {model.coef_.shape}")
    print(f"   ✓ Model training successful\n")
    
    # Make predictions
    print("4. Generating predictions...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Predictions generated: {len(y_pred)} samples")
    print(f"   Test accuracy: {accuracy:.2%}")
    print(f"   ✓ Predictions successful\n")
    
    # Verify basic scikit-learn operations
    print("5. Verifying scikit-learn operations...")
    print(f"   ✓ train_test_split: Working")
    print(f"   ✓ LogisticRegression: Working")
    print(f"   ✓ Model fitting: Working")
    print(f"   ✓ Predictions: Working")
    print(f"   ✓ Metrics calculation: Working\n")
    
    print("="*50)
    print("All scikit-learn tests passed successfully!")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    exit(main())
