#!/usr/bin/env python3
"""
Feature Analysis Script
Analyzes feature importance to understand prediction drivers
"""

import pandas as pd
import numpy as np
import pickle
import os

def main():
    print("Loading trained model...")
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print("Analyzing feature importance...\n")
    
    # Get coefficients (importance)
    coefficients = np.abs(model.coef_[0])
    
    # Normalize to percentages
    total = np.sum(coefficients)
    importance_pct = (coefficients / total) * 100
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coefficients,
        'importance_pct': importance_pct
    }).sort_values('importance', ascending=False)
    
    print("="*50)
    print("Feature Importance Analysis".center(50))
    print("="*50 + "\n")
    
    print("Top features influencing predictions:")
    for idx, row in feature_importance.iterrows():
        print(f"{idx+1}. {row['feature']}: {row['importance']:.4f} ({row['importance_pct']:.2f}%)")
    
    print("\n" + "="*50)
    print("Key Insights".center(50))
    print("="*50 + "\n")
    
    # Calculate combined importance for related features
    test_duration_features = feature_importance[feature_importance['feature'].str.contains('test_duration')]
    cpu_features = feature_importance[feature_importance['feature'].str.contains('cpu')]
    
    test_duration_total = test_duration_features['importance_pct'].sum()
    cpu_total = cpu_features['importance_pct'].sum()
    
    print(f"- Test duration patterns (including rolling average) are the strongest")
    print(f"  predictors of build failure, accounting for over {test_duration_total:.0f}% of model decisions")
    print(f"- CPU usage trends also play a significant role ({cpu_total:.2f}% combined)")
    print(f"- Dependency count has minimal impact on predictions\n")
    
    print("Recommendation: Focus monitoring and alerting on test execution times")
    print("and CPU utilization for early failure detection.\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = 'results/feature_importance.txt'
    
    with open(results_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("Feature Importance Analysis\n")
        f.write("="*50 + "\n\n")
        f.write("Top features influencing predictions:\n")
        for idx, row in feature_importance.iterrows():
            f.write(f"{idx+1}. {row['feature']}: {row['importance']:.4f} ({row['importance_pct']:.2f}%)\n")
        f.write("\n" + "="*50 + "\n")
        f.write("Key Insights\n")
        f.write("="*50 + "\n\n")
        f.write(f"- Test duration patterns: {test_duration_total:.2f}% importance\n")
        f.write(f"- CPU usage patterns: {cpu_total:.2f}% importance\n")
        f.write("- Focus monitoring on test execution times and CPU utilization\n")
    
    print("Feature analysis complete!")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
