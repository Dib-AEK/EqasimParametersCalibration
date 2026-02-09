#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance comparison visualization utilities

This module provides functions to compare distance distributions 
between MZ (Mikrozensus) survey data and MATSim simulation results.

Created on: 2026-02-09
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_distance_distributions_by_purpose(mz_trips, matsim_trips):
    """
    Plot histograms of euclidean distance distributions by trip purpose,
    comparing MZ survey data (weighted) with MATSim simulation results.
    
    Parameters:
    -----------
    mz_trips : pd.DataFrame
        MZ trips DataFrame with columns: 'purpose', 'euclidean_distance_km', 'person_weight'
    matsim_trips : pd.DataFrame
        MATSim trips DataFrame with columns: 'purpose', 'euclidean_distance_km'
    
    Notes:
    ------
    The weight parameter in pandas .plot.hist() expects an array-like object (Series or array),
    not a column name string. This function demonstrates the correct usage.
    """
    for purpose in mz_trips["purpose"].unique():
        mask_mz = (mz_trips["purpose"] == purpose)
        mask_matsim = (matsim_trips["purpose"] == purpose)
        
        bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 35, 500]    
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # FIXED: Pass the actual Series of weights, not the column name string
        # The weights parameter expects array-like data, not a string
        mz_trips.loc[mask_mz, "euclidean_distance_km"].plot.hist(
            ax=ax, bins=bins, density=True, alpha=0.5, label="MZ", 
            weights=mz_trips.loc[mask_mz, "person_weight"]
        )
        
        matsim_trips.loc[mask_matsim, "euclidean_distance_km"].plot.hist(
            ax=ax, bins=bins, density=True, alpha=0.5, label="MATSim"
        )
        
        plt.xlim([0, 50])
        plt.xlabel("Euclidean Distance (km)")
        plt.ylabel("Density")
        plt.legend()
        plt.title(purpose)
        plt.show()


def plot_distance_distributions_by_purpose_v2(mz_trips, matsim_trips, save_path=None):
    """
    Alternative implementation using numpy histogram for more control.
    
    Parameters:
    -----------
    mz_trips : pd.DataFrame
        MZ trips DataFrame with columns: 'purpose', 'euclidean_distance_km', 'person_weight'
    matsim_trips : pd.DataFrame
        MATSim trips DataFrame with columns: 'purpose', 'euclidean_distance_km'
    save_path : str, optional
        Path to save the figures. If None, displays interactively.
    """
    for purpose in mz_trips["purpose"].unique():
        mask_mz = (mz_trips["purpose"] == purpose)
        mask_matsim = (matsim_trips["purpose"] == purpose)
        
        bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 35, 500]    
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Get filtered data
        mz_distances = mz_trips.loc[mask_mz, "euclidean_distance_km"]
        mz_weights = mz_trips.loc[mask_mz, "person_weight"]
        matsim_distances = matsim_trips.loc[mask_matsim, "euclidean_distance_km"]
        
        # Create weighted histogram for MZ data
        ax.hist(mz_distances, bins=bins, density=True, alpha=0.5, 
                label="MZ", weights=mz_weights)
        
        # Create histogram for MATSim data
        ax.hist(matsim_distances, bins=bins, density=True, alpha=0.5, 
                label="MATSim")
        
        ax.set_xlim([0, 50])
        ax.set_xlabel("Euclidean Distance (km)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(purpose)
        
        if save_path:
            plt.savefig(f"{save_path}/distance_dist_{purpose}.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
