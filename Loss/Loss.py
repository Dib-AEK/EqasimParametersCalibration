#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:04:30 2025

@author: dabdelkader
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from Utilities.TourUtility import TourUtility
from Utilities.Selector import Selector
from Utilities.BaseUtility import BaseUtility


class Loss:
    
    def __init__(self, actual_mode_shares: dict, metric = "mse"):
        """
        Parameters:
        - tours: pd.DataFrame with a 'chosen_mode' column (representing selected transport modes)
        - actual_mode_shares: dict, e.g. {'car': 0.4, 'pt': 0.3, 'walk': 0.2, 'bike': 0.1}
        """        
        self.actual = actual_mode_shares
        self.modes = sorted(actual_mode_shares.keys())   
        self.metric = metric
        
    def get_loss(self, parameters=None):
        if parameters is not None:
            BaseUtility.set_parameters(parameters)
        return self._get_loss()
    
    
    @property
    def estimated_mode_shares(self):
        return self.get_estimated_mode_shares()
    
    def get_estimated_mode_shares(self, modes = None):
        """
        Computes mode shares from the tours.
        Returns:
            dict of estimated shares (same keys as actual)
        """
        if modes == None:
            modes = self.modes
            
        tours = TourUtility.get_all_utilities()
        tours = Selector.get_mode_shares_from_tours(tours)
        
        selected_modes = tours.loc[tours.selected,"candidate_mode"].explode()
        
        counts = selected_modes.value_counts(normalize=True)
        estimated = {mode: counts.get(mode, 0.0) for mode in modes}
        return estimated
    
    @property
    def actual_mode_shares(self):
        return self.get_actual_mode_shares()
    
    def get_actual_mode_shares(self, modes = None):
        """
        Return the actual mode shares (The observed ones, used for calibration).
        Returns:
            dict of estimated shares (same keys as actual)
        """
        if modes == None:
            modes = self.modes            
        actual = {mode: self.actual.get(mode, 0.0) for mode in modes}
        return actual
    
    @property
    def eqasim_mode_shares(self):
        return self.get_eqasim_mode_shares()
    
    def get_eqasim_mode_shares(self, modes = None):
        """
        Computes mode shares that are selected in eqasim (DMC) at that specific iteration.
        Returns:
            dict of estimated shares (same keys as actual)
        """
        if modes == None:
            modes = self.modes            
        
        selected_modes = TourUtility.tours.loc[TourUtility.tours["eqasim_selected"], "candidate_mode"].explode()
        counts         = selected_modes.value_counts(normalize=True)
        actual         = {mode: counts.get(mode, 0.0) for mode in modes}
        return actual
    
    def _get_loss(self):
        metric = self.metric.lower()
        
        if metric == "mse":
            return self.mse()
        elif metric == "mae":
            return self.mae()
        elif metric == "cosine" or metric == "cosine_similarity":
            return self.cosine_similarity()
        elif metric == "kl" or metric == "kl_divergence":
            return self.kl_divergence()
        elif metric == "js" or metric == "js_divergence":
            return self.js_divergence()
        elif metric == "hellinger" or metric == "hellinger_distance":
            return self.hellinger_distance()
        elif metric == "tv" or metric == "total_variation" or metric == "total_variation_distance" or metric == "l_distance":
            return self.total_variation_distance()
        else:
            raise ValueError(f"Unknown loss metric: '{self.metric}'")
    

    def _vectors(self):
        """
        Returns actual and estimated mode shares as aligned numpy arrays.
        """
        est = self.estimated_mode_shares
        actual_vec = np.array([self.actual[mode] for mode in self.modes])
        est_vec = np.array([est[mode] for mode in self.modes])
        return actual_vec, est_vec

    def mse(self):
        actual, est = self._vectors()
        return mean_squared_error(actual, est)

    def mae(self):
        actual, est = self._vectors()
        return mean_absolute_error(actual, est)

    def cosine_similarity(self):
        actual, est = self._vectors()
        return cosine_similarity([actual], [est])[0, 0]
    
    def kl_divergence(self):
        actual, est = self._vectors()
        return np.sum(rel_entr(actual, est + 1e-12)) 
    
    def js_divergence(self):
        actual, est = self._vectors()
        return jensenshannon(actual, est, base=2)
    
    def hellinger_distance(self):
        actual, est = self._vectors()
        return np.sqrt(np.sum((np.sqrt(actual) - np.sqrt(est)) ** 2)) / np.sqrt(2)
    
    def total_variation_distance(self):
        actual, est = self._vectors()
        return 0.5 * np.sum(np.abs(actual - est))

    def summary(self):
        return {
            "mse": self.mse(),
            "mae": self.mae(),
            "cosine_similarity": self.cosine_similarity(),
            "kl_divergence": self.kl_divergence(),
            "js_divergence": self.js_divergence(),
            "hellinger_distance": self.hellinger_distance(),
            "total_variation_distance": self.total_variation_distance()
        }
