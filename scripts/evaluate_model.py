#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates the trained model using confusion matrix and classification metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
import os

def main():
    print("Loading test data...")
    test_df = pd.read_csv('data/test_data.csv')
    print(f"Test set: {len(test_df)} samples")
    
    # Prepare features and target
    X_test = test_df.drop('build_status', axis=1)
    y_test = (test_df['build_status'] == 'success').astype(int)
    
    print("Loading trained model...")
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Model Evaluation Results".center(50))
    print("="*50 + "\n")
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Display confusion matrix with labels
    print("Confusion Matrix:")
    print(f"{'':20} Predicted Failure  Predicted Success")
    print(f"Actual Failure   {cm[0][0]:17d} {cm[0][1]:17d}")
    print(f"Actual Success   {cm[1][0]:17d} {cm[1][1]:17d}\n")
    
    # Classification report
    print("Classification Report:")
    target_names = ['Failure', 'Success']
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # Interpretation
    print("="*50)
    print("Interpretation".center(50))
    print("="*50 + "\n")
    
    tn, fp, fn, tp = cm[0][0], cm[1][0], cm[0][1], cm[1][1]
    print(f"True Positives (Correctly predicted failures): {tn}")
    print(f"False Positives (Incorrectly predicted failures): {fp}")
    print(f"True Negatives (Correctly predicted successes): {tp}")
    print(f"False Negatives (Missed failures): {fn}\n")
    
    recall_failure = tn / (tn + fn) if (tn + fn) > 0 else 0
    print(f"The model correctly identifies {recall_failure*100:.0f}% of actual build failures.")
    print(f"However, it misses {fn} failures ({(fn/(tn+fn))*100:.0f}% false negative rate).")
    print("In a production environment, these missed failures could lead to")
    print("unexpected issues being deployed.\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = 'results/evaluation_report.txt'
    
    with open(results_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("Model Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{'':20} Predicted Failure  Predicted Success\n")
        f.write(f"Actual Failure   {cm[0][0]:17d} {cm[0][1]:17d}\n")
        f.write(f"Actual Success   {cm[1][0]:17d} {cm[1][1]:17d}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "="*50 + "\n")
        f.write("Interpretation\n")
        f.write("="*50 + "\n\n")
        f.write(f"True Positives: {tn}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"True Negatives: {tp}\n")
        f.write(f"False Negatives: {fn}\n")
    
    print("Evaluation complete!")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
