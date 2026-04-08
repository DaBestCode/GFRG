import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def compute_mmd(X, Y, gamma=None):
    """
    Computes the Maximum Mean Discrepancy (MMD). 
    Higher MMD = Larger Distribution Shift.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1] 
        
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    
    return XX.mean() + YY.mean() - 2 * XY.mean()

def main():
    print("Loading Boston Housing dataset...")
    boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
    
    # Drop the target variable (we only measure feature shift, not label shift)
    df = boston.frame.drop(columns=['MEDV']) 
    
    # --- LOAD THE EXACT SPLIT THE RL AGENT USES ---
    base = os.path.dirname(os.path.abspath(__file__))
    idx_path = os.path.join(base, 'data/processed/environments/current_env_index.npy')
    
    if not os.path.exists(idx_path):
        print(f"Error: Could not find {idx_path}. Please run split_environments.py first.")
        return
        
    env_sort_index = np.load(idx_path)
    sorted_data = df.loc[env_sort_index].copy()
    
    # Normalize the data (Crucial for distance metrics)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sorted_data)
    
    # Apply the 35% / 65% split
    total_rows = len(X_scaled)
    split_1 = int(total_rows * 0.35)
    split_2 = int(total_rows * 0.65)
    
    env_A = X_scaled[:split_1]
    env_B = X_scaled[split_2:]
    
    # Calculate Scores
    mmd_AB = compute_mmd(env_A, env_B)
    mmd_AA = compute_mmd(env_A, env_A)
    
    print("\n" + "="*40)
    print(f"DISTRIBUTION SHIFT SCORE (MMD)")
    print("="*40)
    print(f"Sanity Check (Env A vs Env A): {mmd_AA:.5f} (Should be ~0)")
    print(f"Actual Shift (Env A vs Env B): {mmd_AB:.5f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()