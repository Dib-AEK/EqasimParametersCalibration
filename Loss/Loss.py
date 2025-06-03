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
from modeShares.modeShares import ModeShares

class Loss:
    
    def __init__(self, mode_shares_provider: ModeShares,
                       metric = "js", 
                       calibrate_global_modeshare:bool=True, 
                       calibrate_modeshare_distribution:bool=False):
        """
        Parameters:
        - tours: pd.DataFrame with a 'chosen_mode' column (representing selected transport modes)
        - actual_mode_shares: dict, e.g. {'car': 0.4, 'pt': 0.3, 'walk': 0.2, 'bike': 0.1}
        - mode_shares_distribution: dict,similar to actual_mode_shares but with a list 
        for each mode and an extra key (distance)
        """        
        self.actual_global_mode_shares = mode_shares_provider.get_mode_shares()
        self.actual_mode_shares_distribution = mode_shares_provider.get_mode_shares_distribution()
        self.distance_bins = mode_shares_provider.get_distance_bins()
        self.distance_labels = mode_shares_provider.get_distance_labels()
        
        self.calibrate_global_modeshare = calibrate_global_modeshare
        self.calibrate_modeshare_distribution = calibrate_modeshare_distribution
        self.metric = metric
        
        self.modes = ["car","walk","bike","pt","car_passenger"]         
        self.calibration_modes = ["car","pt","bike","walk"]
        
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
        tours = Selector.select(tours)
        
        cols = ["candidate_mode", "euclidean_distance"] 
        selected_modes = tours.loc[tours.selected, cols].explode(column = cols)
        selected_modes = selected_modes[selected_modes.euclidean_distance>1e-3] #same selection as in modeShares
        
        estimates_global_mode_share = dict()
        estimates_mode_share_distribution = dict()
        if self.calibrate_global_modeshare:
            counts = selected_modes["candidate_mode"].value_counts(normalize=True)
            estimates_global_mode_share = {mode: counts.get(mode, 0.0) for mode in modes}
        
        if self.calibrate_modeshare_distribution:
            distance_bins = np.array(self.distance_bins)*1e-3 #convert to km
            bin_labels    =  self.distance_labels
            selected_modes['distance_bin'] = pd.cut(selected_modes['euclidean_distance'],
                                                    bins=distance_bins,
                                                    labels=bin_labels, 
                                                    include_lowest=True, 
                                                    ordered=True)
            grouped = selected_modes.groupby(['distance_bin', 'candidate_mode'], observed=False).size().unstack(fill_value=0)
            mode_shares_by_bin = grouped.div(grouped.sum(axis=1), axis=0).fillna(0)
            estimates_mode_share_distribution = {mode: mode_shares_by_bin[mode].tolist()
                                                 for mode in modes}
                
        return estimates_global_mode_share, estimates_mode_share_distribution
    
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
        actual = {mode: self.actual_global_mode_shares.get(mode, 0.0) for mode in modes}        
        actual_distribution = {mode: self.actual_mode_shares_distribution.get(mode, 0.0) for mode in modes}
        return actual, actual_distribution
    
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
        cols = ["candidate_mode","euclidean_distance"] 
        selected_modes = TourUtility.tours.loc[TourUtility.tours["eqasim_selected"], cols].explode()
        selected_modes = selected_modes[selected_modes.euclidean_distance>1e-3] #same selection as in modeShares
        
        eqasim_global_mode_share = dict()        
        eqasim_mode_share_distribution = dict()
        if self.calibrate_global_modeshare:
            counts = selected_modes["candidate_mode"].value_counts(normalize=True)
            eqasim_global_mode_share = {mode: counts.get(mode, 0.0) for mode in modes}
            
        if self.calibrate_modeshare_distribution:
            distance_bins = self.actual_mode_shares_distribution["distance"]
            bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}" for i in range(len(distance_bins)-1)]
            selected_modes['distance_bin'] = pd.cut(selected_modes['euclidean_distance'],
                                                    bins=distance_bins,
                                                    labels=bin_labels,
                                                    include_lowest=True,
                                                    right=True)
            grouped = selected_modes.groupby(['distance_bin', 'candidate_mode'], observed=False).size().unstack(fill_value=0)
            mode_shares_by_bin = grouped.div(grouped.sum(axis=1), axis=0).fillna(0)
            eqasim_mode_share_distribution = {mode: mode_shares_by_bin[mode].tolist()
                                                 for mode in modes}
        
        return eqasim_global_mode_share, eqasim_mode_share_distribution
    
    def _get_loss(self):
        metric = self.metric.lower()
        func = self.get_loss_func(metric)
        actual, est, actual_dist, est_dist = self._vectors()
        loss = 0
        if self.calibrate_global_modeshare:
            loss += func(actual, est)
        if self.calibrate_modeshare_distribution:
            loss += func(actual_dist, est_dist)
        return loss

    def _vectors(self):
        """
        Returns actual and estimated mode shares as aligned numpy arrays.
        """
        est,dist = self.estimated_mode_shares
        actual,actual_dist = self.actual_mode_shares
        calibration_modes = self.calibration_modes
                
        est_vec, est_dist_vec, actual_vec, actual_dist_vec = None, None, None, None
        if self.calibrate_global_modeshare:
            actual_vec = np.array([actual[mode] for mode in calibration_modes])            
            est_vec = np.array([est[mode] for mode in calibration_modes])            
        
        if self.calibrate_modeshare_distribution:            
            actual_dist_vec = np.array([actual_dist[mode] for mode in calibration_modes]).flatten()
            est_dist_vec = np.array([dist[mode] for mode in calibration_modes]).flatten()
        
        return actual_vec, est_vec, actual_dist_vec, est_dist_vec
    
    def get_loss_func(self, metric):
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
    
    def mse(self):        
        return mean_squared_error

    def mae(self):
        return mean_absolute_error

    def cosine_similarity(self):
        return lambda x,y: cosine_similarity([x], [y])[0, 0]
        
    def kl_divergence(self):
        return lambda x,y: np.sum(rel_entr(x, y + 1e-12))  
    
    def js_divergence(self):
        return jensenshannon
    
    def hellinger_distance(self):
        return  lambda x,y: np.sqrt(np.sum((np.sqrt(x) - np.sqrt(y)) ** 2)) / np.sqrt(2)
    
    def total_variation_distance(self):
        return lambda x,y: np.sum(np.abs(x - y))

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
