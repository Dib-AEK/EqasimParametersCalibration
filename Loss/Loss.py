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
from Selector.Selector import Selector
from Utilities.BaseUtility import BaseUtility
from modeShares.modeShares import ModeShares
import time
import polars as pl
from Loss.Losses import Losses
import Loss.utils as U



POSSIBLE_OBJECTIVES = ["global","distance","canton","age","income","sp_region",
                       "mode_distance", "mode_income","mode_age","mode_canton"]

ATTR_TO_COL = {"age":"age_class","income":'income_class',"canton":"canton_id",
               "distance":"distance", "sp_region":"sp_region"}

WEIGHTS = {"global":1.0,"distance":1.0,"canton":0.5,"age":0.3,"income":0.5, 
           "sp_region":2.0,#because the list is short, only 3 regions
           "mode_distance":0.8, "mode_income":0.5,"mode_age":0.3,"mode_canton":0.5}

WEIGHT_MODE = {"pt":1.0,"car":0.8,"walk":2.0,"bike":2.0,"car_passenger":1.0}#because walk and bike all zeros for long trips

class Loss(Losses):
    
    def __init__(self, mode_shares_provider: ModeShares,
                       metric = "js", 
                       objectives:list = None,
                       calibration_modes:list = ["car","pt","bike","walk"]):
          
        super().__init__(metric)
        
        self._actual_mode_shares = mode_shares_provider.get_mode_shares()                
        self.distance_bins = mode_shares_provider.get_distance_bins()
        self.distance_labels = mode_shares_provider.get_distance_labels()               
        
        self.objectives = objectives
        if objectives is not None:           
            assert all([obj in POSSIBLE_OBJECTIVES for obj in objectives])
        else:
            self.objectives = ["global", "distance"]
        
        self.modes = ["car","walk","bike","pt","car_passenger"]   #Should be all simulated modes       
        self.calibration_modes = calibration_modes
        self.calibrated_modes_weights = np.array([WEIGHT_MODE[mode] for mode in calibration_modes]).reshape(-1,1)
        self.utility_time = []
        self.selector_time = []
        self.mode_share_time = []

    def get_actual_mode_shares(self):
        """
        Returns the actual (observed) mode shares used for calibration.
    
        Args:
            modes (list, optional): List of modes to return shares for. Defaults to self.modes.
    
        Returns:
                - dict of mode shares
        """
        modes = self.modes
        # Global and distributional actual shares (estimate it only one time for better efficiency)   
        if not hasattr(self, "_organized_mode_shares"):            
            self._organized_mode_shares =  {
                "global": U.get_shares(self._actual_mode_shares["global"], modes),
                "distance": U.get_shares(self._actual_mode_shares["distance"], modes),
                "age":      U.get_shares(self._actual_mode_shares["age"], modes),
                "income":   U.get_shares(self._actual_mode_shares["income"], modes),
                "canton":   U.get_shares(self._actual_mode_shares["canton"], modes),
                "mode_distance": U.get_shares(self._actual_mode_shares["mode_distance"], modes),
                "mode_age": U.get_shares(self._actual_mode_shares["mode_age"], modes),
                "mode_income": U.get_shares(self._actual_mode_shares["mode_income"], modes),
                "mode_canton": U.get_shares(self._actual_mode_shares["mode_canton"], modes),
                "sp_region": U.get_shares(self._actual_mode_shares["sp_region"], modes),
                    }
    
        return self._organized_mode_shares
    
    def get_estimated_mode_shares(self):
        """
        Computes estimated mode shares from the tours.
        
        Returns:
            - dict of mode shares
        """
        # Use default modes if not provided        
        modes = self.modes 
        objectives = self.objectives
        mode_shares = {}
        
        # Retrieve and filter tours
        tours = TourUtility.get_all_utilities().collect()
        tours = Selector.select(tours)
    
        # Select and explode relevant columns
        base_columns = ["candidate_mode", "euclidean_distance", "income_class", 
                        "canton_id", "age_class", "sp_region"]
        exploded_columns = ["candidate_mode", "euclidean_distance"]
        selected_modes = tours.select(base_columns).explode(exploded_columns).filter(pl.col("euclidean_distance") > 1e-3)
        
        # TODO: add distance bins in base columns
        if "distance" in objectives or "mode_distance" in objectives:
            distance_bins = np.array(self.distance_bins)*1e-3 #convert to km    
            selected_modes = selected_modes.with_columns([
                pl.col("euclidean_distance").cut(breaks=distance_bins[1:-1], 
                                                 labels=self.distance_labels).alias("distance")])
            
        # get the mode shares        
        if "global" in objectives:
            mode_shares["global"] = U.compute_global_mode_share(selected_modes, modes)   
                                        
        for k,v in ATTR_TO_COL.items():
            if k in objectives:
                mode_shares[k] = U.compute_mode_share_distribution_by(selected_modes, modes, by=v)
            if f'mode_{k}' in objectives:
                mode_shares[f'mode_{k}'] = U.compute_mode_distribution_by(selected_modes, modes, by=v)            
                                     
        return mode_shares

    def _vectors(self):
        """
        Constructs vectors of actual and estimated mode shares for each objective.
    
        Returns:
            dict: A dictionary mapping each objective to a tuple of NumPy arrays:
                  (actual_mode_shares, estimated_mode_shares). Each array is
                  ordered according to `self.calibration_modes`.
        """
        actual = self.actual_mode_shares
        estimated = self.estimated_mode_shares        
        modes = self.calibration_modes
        objectives = self.objectives        
        output = {}                
        
        for obj in objectives:
            actual_obj = actual[obj]
            estimated_obj = estimated[obj]            
                
            ignore = -1 if "distance" in obj else None
            estimated_vec = np.array([estimated_obj[mode][:ignore] for mode in modes]) # Explicit access by mode to raise an error if a mode is missing
            actual_vec = np.array([actual_obj[mode][:ignore] for mode in modes])
    
            output[obj] = (actual_vec, estimated_vec)
            # print(obj, "->", actual_vec.shape)
            
        return output
    
    def _get_loss(self):
        """
        Compute the total calibration loss across global and distributional mode shares.
    
        Returns:
            float: Total loss scaled by 100.0
        """
        metric = self.metric.lower()
        loss_func = self.get_loss_func(metric)
        mode_weights = self.calibrated_modes_weights
        
        vectors = self._vectors()    
        loss = 0.0
        total_weights = 0.0
        
        if "global" in self.objectives:
            assert mode_weights.shape==vectors["global"][0].shape
            
        for k,(actual,estimated) in vectors.items():
            weight = WEIGHTS[k]
            x = (actual*mode_weights).flatten()
            y = (estimated*mode_weights).flatten()
            loss += weight*loss_func(x, y)
            total_weights += weight
              
        return np.log(loss/total_weights) # I use the log because it highlights better the minima




    
    