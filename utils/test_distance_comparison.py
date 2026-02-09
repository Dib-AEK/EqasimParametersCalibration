#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for distance_comparison utilities

Tests the corrected weight parameter usage in histogram plotting.
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from utils.distance_comparison import (
    plot_distance_distributions_by_purpose,
    plot_distance_distributions_by_purpose_v2
)


class TestDistanceComparison(unittest.TestCase):
    """Test cases for distance comparison utilities"""
    
    def setUp(self):
        """Create sample data for testing"""
        np.random.seed(42)
        
        # Create sample MZ trips with weights
        self.mz_trips = pd.DataFrame({
            'purpose': np.random.choice(['work', 'shopping', 'leisure'], size=500),
            'euclidean_distance_km': np.abs(np.random.normal(10, 5, size=500)),
            'person_weight': np.random.uniform(0.5, 2.0, size=500)
        })
        
        # Create sample MATSim trips (no weights)
        self.matsim_trips = pd.DataFrame({
            'purpose': np.random.choice(['work', 'shopping', 'leisure'], size=500),
            'euclidean_distance_km': np.abs(np.random.normal(10, 5, size=500))
        })
    
    def test_plot_with_correct_weights(self):
        """Test that plotting with correct weights parameter works"""
        try:
            # This should work without errors
            purpose = 'work'
            mask_mz = (self.mz_trips["purpose"] == purpose)
            mask_matsim = (self.matsim_trips["purpose"] == purpose)
            
            bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 35, 500]
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # CORRECT usage
            self.mz_trips.loc[mask_mz, "euclidean_distance_km"].plot.hist(
                ax=ax, bins=bins, density=True, alpha=0.5, label="MZ",
                weights=self.mz_trips.loc[mask_mz, "person_weight"]
            )
            
            self.matsim_trips.loc[mask_matsim, "euclidean_distance_km"].plot.hist(
                ax=ax, bins=bins, density=True, alpha=0.5, label="MATSim"
            )
            
            plt.close()
            
        except Exception as e:
            self.fail(f"Correct weights usage should not raise exception: {e}")
    
    def test_plot_with_incorrect_weights_fails(self):
        """Test that incorrect weight parameter raises an error"""
        purpose = 'work'
        mask_mz = (self.mz_trips["purpose"] == purpose)
        
        bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 35, 500]
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # INCORRECT usage should fail
        with self.assertRaises((AttributeError, TypeError)):
            self.mz_trips.loc[mask_mz, "euclidean_distance_km"].plot.hist(
                ax=ax, bins=bins, density=True, alpha=0.5, label="MZ",
                weight="person_weight"  # Wrong: string instead of Series
            )
        
        plt.close()
    
    def test_weights_series_length_matches_data(self):
        """Test that weights Series has same length as filtered data"""
        purpose = 'work'
        mask_mz = (self.mz_trips["purpose"] == purpose)
        
        distances = self.mz_trips.loc[mask_mz, "euclidean_distance_km"]
        weights = self.mz_trips.loc[mask_mz, "person_weight"]
        
        self.assertEqual(len(distances), len(weights),
                        "Weights must have same length as data")
    
    def test_weights_are_numeric(self):
        """Test that weights are numeric values"""
        weights = self.mz_trips["person_weight"]
        
        self.assertTrue(pd.api.types.is_numeric_dtype(weights),
                       "Weights must be numeric")
        self.assertTrue(all(weights > 0),
                       "Weights should be positive")
    
    def test_alternative_hist_method(self):
        """Test alternative histogram method using plt.hist directly"""
        try:
            purpose = 'work'
            mask_mz = (self.mz_trips["purpose"] == purpose)
            mask_matsim = (self.matsim_trips["purpose"] == purpose)
            
            bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 35, 500]
            fig, ax = plt.subplots(figsize=(10, 5))
            
            mz_distances = self.mz_trips.loc[mask_mz, "euclidean_distance_km"]
            mz_weights = self.mz_trips.loc[mask_mz, "person_weight"]
            matsim_distances = self.matsim_trips.loc[mask_matsim, "euclidean_distance_km"]
            
            ax.hist(mz_distances, bins=bins, density=True, alpha=0.5,
                   label="MZ", weights=mz_weights)
            ax.hist(matsim_distances, bins=bins, density=True, alpha=0.5,
                   label="MATSim")
            
            plt.close()
            
        except Exception as e:
            self.fail(f"Alternative histogram method should work: {e}")


class TestUtilityFunctions(unittest.TestCase):
    """Test the utility functions in distance_comparison module"""
    
    def setUp(self):
        """Create sample data for testing"""
        np.random.seed(42)
        
        self.mz_trips = pd.DataFrame({
            'purpose': ['work'] * 100 + ['shopping'] * 100,
            'euclidean_distance_km': np.abs(np.random.normal(10, 5, size=200)),
            'person_weight': np.random.uniform(0.5, 2.0, size=200)
        })
        
        self.matsim_trips = pd.DataFrame({
            'purpose': ['work'] * 100 + ['shopping'] * 100,
            'euclidean_distance_km': np.abs(np.random.normal(10, 5, size=200))
        })
    
    def test_plot_function_v2_with_save(self):
        """Test plot function with save_path parameter"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                plot_distance_distributions_by_purpose_v2(
                    self.mz_trips, self.matsim_trips, save_path=tmpdir
                )
                
                # Check that files were created
                files = os.listdir(tmpdir)
                self.assertGreater(len(files), 0, "Should create plot files")
                
                # Check file naming
                for f in files:
                    self.assertTrue(f.startswith('distance_dist_'),
                                  f"File should start with 'distance_dist_': {f}")
                    self.assertTrue(f.endswith('.png'),
                                  f"File should end with '.png': {f}")
            
            except Exception as e:
                self.fail(f"Plot function v2 should work: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
