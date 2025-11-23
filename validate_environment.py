#!/usr/bin/env python3
"""
Environment Validation Script
Verifies that all required Python ML libraries are installed and working
"""

import sys

def main():
    try:
        import pandas as pd
        import sklearn
        import numpy as np
        
        print("Environment validation successful!")
        print(f"Pandas version: {pd.__version__}")
        print(f"Scikit-learn version: {sklearn.__version__}")
        print(f"NumPy version: {np.__version__}")
        sys.exit(0)
    except ImportError as e:
        print(f"Environment validation failed: {e}")
        print("\nPlease ensure all required libraries are installed:")
        print("  - pandas")
        print("  - scikit-learn")
        print("  - numpy")
        sys.exit(1)

if __name__ == "__main__":
    main()
