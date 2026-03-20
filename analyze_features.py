import pandas as pd
import sys
import os

print("=== The AI4SPACE Feature Decoder ===")

if len(sys.argv) < 2:
    print("Usage: python analyze_features.py [path_to_optimal_hdf]")
    sys.exit(1)

optimal_file = sys.argv[1]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
original_file = os.path.join(BASE_DIR, 'data/processed/housing_boston.hdf')

try:
    # Load both datasets
    df_opt = pd.read_hdf(optimal_file)
    df_orig = pd.read_hdf(original_file)
    
    orig_cols = set(df_orig.columns)
    opt_cols = set(df_opt.columns)
    
    # Isolate the changes
    kept_original = orig_cols.intersection(opt_cols)
    dropped_original = orig_cols - opt_cols
    new_generated = opt_cols - orig_cols
    
    print(f"\nAnalyzing: {optimal_file}")
    print(f"Total Features in Optimized Set: {len(opt_cols)}")
    
    print("\n[-] ORIGINAL FEATURES DROPPED (Deemed not robust):")
    if not dropped_original:
        print("  (None)")
    for col in dropped_original:
        print(f"  - Column {col}")
        
    print("\n[+] NEW FEATURES GENERATED (The IRM Survivors):")
    if not new_generated:
        print("  (None)")
    for col in new_generated:
        print(f"  - {col}")

except Exception as e:
    print(f"Error loading files: {e}")
