#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Distance Distribution Comparison

This example demonstrates how to correctly compare distance distributions
between MZ (Mikrozensus) survey data and MATSim simulation results.

The key fix addresses the weight parameter issue in histogram plotting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def example_distance_comparison():
    """
    Example showing the CORRECT way to create weighted histograms
    for comparing MZ survey data with MATSim simulation results.
    """
    
    # Load your data (example paths - adjust as needed)
    # mz_trips = pd.read_csv("path/to/mz_trips.csv")
    # matsim_trips = pd.read_csv("path/to/matsim_trips.csv")
    
    # For demonstration, create sample data
    np.random.seed(42)
    mz_trips = pd.DataFrame({
        'purpose': np.random.choice(['work', 'shopping', 'leisure'], size=1000),
        'euclidean_distance_km': np.abs(np.random.normal(10, 5, size=1000)),
        'person_weight': np.random.uniform(0.5, 2.0, size=1000)
    })
    
    matsim_trips = pd.DataFrame({
        'purpose': np.random.choice(['work', 'shopping', 'leisure'], size=1000),
        'euclidean_distance_km': np.abs(np.random.normal(10, 5, size=1000))
    })
    
    # CORRECTED CODE - Method 1: Using pandas .plot.hist()
    print("Generating histograms using corrected code...")
    for purpose in mz_trips["purpose"].unique():
        mask_mz = (mz_trips["purpose"] == purpose)
        mask_matsim = (matsim_trips["purpose"] == purpose)
        
        bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 35, 500]    
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # CORRECT: Pass the actual Series of weights, not the column name
        mz_trips.loc[mask_mz, "euclidean_distance_km"].plot.hist(
            ax=ax, bins=bins, density=True, alpha=0.5, label="MZ", 
            weights=mz_trips.loc[mask_mz, "person_weight"]  # ← FIXED HERE
        )
        
        matsim_trips.loc[mask_matsim, "euclidean_distance_km"].plot.hist(
            ax=ax, bins=bins, density=True, alpha=0.5, label="MATSim"
        )
        
        plt.xlim([0, 50])
        plt.xlabel("Euclidean Distance (km)")
        plt.ylabel("Density")
        plt.legend()
        plt.title(f"Distance Distribution - {purpose}")
        plt.grid(True, alpha=0.3)
        
        # Save or show
        plt.savefig(f'/tmp/example_method1_{purpose}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Created histogram for purpose: {purpose}")
    
    # Alternative Method 2: Using matplotlib.pyplot.hist directly (recommended)
    print("\nGenerating histograms using alternative method...")
    for purpose in mz_trips["purpose"].unique():
        mask_mz = (mz_trips["purpose"] == purpose)
        mask_matsim = (matsim_trips["purpose"] == purpose)
        
        bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 35, 500]    
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Extract data explicitly (clearer and more maintainable)
        mz_distances = mz_trips.loc[mask_mz, "euclidean_distance_km"]
        mz_weights = mz_trips.loc[mask_mz, "person_weight"]
        matsim_distances = matsim_trips.loc[mask_matsim, "euclidean_distance_km"]
        
        # Create histograms
        ax.hist(mz_distances, bins=bins, density=True, alpha=0.5, 
                label="MZ (weighted)", weights=mz_weights, color='blue')
        ax.hist(matsim_distances, bins=bins, density=True, alpha=0.5, 
                label="MATSim", color='orange')
        
        ax.set_xlim([0, 50])
        ax.set_xlabel("Euclidean Distance (km)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"Distance Distribution - {purpose}", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f'/tmp/example_method2_{purpose}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Created histogram for purpose: {purpose}")
    
    print("\n✓ All histograms generated successfully!")
    print("  Saved to /tmp/example_method*.png")


def summary_of_fix():
    """Print a summary of the fix"""
    print("=" * 70)
    print("SUMMARY OF THE FIX")
    print("=" * 70)
    print()
    print("INCORRECT (Original Code):")
    print("  weight='person_weight'")
    print("  ❌ Wrong parameter name (should be 'weights', plural)")
    print("  ❌ Wrong type (string instead of array-like)")
    print()
    print("CORRECT (Fixed Code):")
    print("  weights=mz_trips.loc[mask_mz, 'person_weight']")
    print("  ✓ Correct parameter name ('weights')")
    print("  ✓ Correct type (pandas Series)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    summary_of_fix()
    print()
    example_distance_comparison()
