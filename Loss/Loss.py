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
from modeShares.ModeShares import ModeShares
import time
import polars as pl
from Loss.Losses import Losses
import Loss.utils as U
import matplotlib.pyplot as plt


POSSIBLE_OBJECTIVES = ["global","distance","canton","age","income","sp_region",
                       "mode_distance", "mode_income","mode_age","mode_canton", 'vot']

ATTR_TO_COL = {"age":"age_class","income":'income_class',"canton":"canton_id",
               "distance":"distance_class", "sp_region":"sp_region"}

WEIGHTS = {"global":2.5,"distance":1.0, "mode_distance":1,
           "vot":1/5e5,
           "canton":0.5,"age":0.3,"income":0.5, "sp_region":1.0, 
            "mode_income":0.5,"mode_age":0.3,"mode_canton":0.5}

WEIGHT_MODE = {"pt":1.0,"car":1.0,"walk":2.0,"bike":2.5,"car_passenger":1.0} #because walk and bike all zeros for long trips, used only for distance distributions

SELECTED_MODE = {"sp_region":['car','pt']} #must be dict of list, for each attributes, if we want to include only some of the modes, not all of them

VoT = {"car":[30.6], "walk":[26.7], "bike":[18.2], "pt":[14.8]}
#VoT are obtained from https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/491385/ab1637.pdf?sequence=2&isAllowed=y
#The value of travel time savings and the value of leisure in Zurich: Estimation, decomposition and policy implications
WEIGHT_MODE_VOT = {"pt":1.0,"car":1.0,"walk":0.1,"bike":0.1,"car_passenger":0.1}


class Loss(Losses):
    
    def __init__(self, mode_shares_provider: ModeShares,
                       metric = "mse", 
                       objectives:list = None):
          
        super().__init__(metric)
        
        self._actual_mode_shares = mode_shares_provider.get_mode_shares()                
        self.distance_bins = mode_shares_provider.get_distance_bins()
        self.distance_labels = mode_shares_provider.get_distance_labels()               
        
        self.objectives = objectives
        if objectives is not None:           
            assert all([obj in POSSIBLE_OBJECTIVES for obj in objectives])
        else:
            self.objectives = ["global", "distance", "mode_distance"]
        
        self.modes = ["car","walk","bike","pt","car_passenger"]   #Should be all simulated modes       
        self.calibration_modes = ["car","pt","bike","walk"]  #calibration modes are only the modes whose losses are considered as objective to minimize
        self.calibrated_modes_weights = np.array([WEIGHT_MODE[mode] for mode in self.calibration_modes]).reshape(-1,1)
        self.calibrated_modes_weights_vot = np.array([WEIGHT_MODE_VOT[mode] for mode in self.calibration_modes]).reshape(-1,1)
        

        self.losses_record = {key: [] for key in objectives}
        
    def _get_vot(self):
        vot_car  = 60 * BaseUtility.car.betaTravelTime_u_min / BaseUtility.cost.betaCost_u_MU
        vot_pt   = 60 * BaseUtility.pt.betaInVehicleTime_u_min / BaseUtility.cost.betaCost_u_MU
        vot_walk = 60 * BaseUtility.walk.betaTravelTime_u_min / BaseUtility.cost.betaCost_u_MU
        vot_bike = 60 * BaseUtility.bike.betaTravelTime_u_min / BaseUtility.cost.betaCost_u_MU
        return dict(car=[vot_car], pt=[vot_pt], walk=[vot_walk], bike=[vot_bike])
    
    def get_actual_mode_shares(self, modes = None):
        """
        Returns the actual (observed) mode shares used for calibration.
    
        Args:
            modes (list, optional): List of modes to return shares for. Defaults to self.modes.
    
        Returns:
                - dict of mode shares
        """
        if modes is None:            
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
                "vot": U.get_shares(VoT, modes),
                    }
    
        return self._organized_mode_shares
    
    def get_estimated_mode_shares(self, modes = None):
        """
        Computes estimated mode shares from the tours.
        
        Returns:
            - dict of mode shares
        """
        # Use default modes if not provided        
        if modes is None:
            modes = self.modes 
            
        objectives = self.objectives
        mode_shares = {}
        
        # Retrieve and filter tours
        tours = TourUtility.get_all_utilities().collect()
        tours = Selector.select(tours)
    
        # Select and explode relevant columns
        base_columns = ["candidate_mode", "euclidean_distance", "income_class", 
                        "canton_id", "age_class", "sp_region", "distance_class"]
        exploded_columns = ["candidate_mode", "euclidean_distance", "distance_class"]
        selected_modes = (tours.select(base_columns)
                               .explode(exploded_columns)
                               .filter(pl.col("euclidean_distance") > 1e-3))
                    
        # get the mode shares        
        if "global" in objectives:
            mode_shares["global"] = U.compute_global_mode_share(selected_modes, modes)   
                                        
        for k,v in ATTR_TO_COL.items():
            if k in objectives:
                mode_shares[k] = U.compute_mode_share_distribution_by(selected_modes, modes, by=v)
            if f'mode_{k}' in objectives:
                mode_shares[f'mode_{k}'] = U.compute_mode_distribution_by(selected_modes, modes, by=v)            

        if 'vot' in objectives:
            mode_shares["vot"] = self._get_vot()
                            
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
    
    def _get_mode_weights(self, obj):
        return self.calibrated_modes_weights if 'vot' not in obj else self.calibrated_modes_weights_vot
                
    def _get_loss(self):
        """
        Compute the total calibration loss across global and distributional mode shares.
    
        Returns:
            float: Total loss scaled by log to highligh minima
        """
        metric = self.metric.lower()
        loss_func = self.get_loss_func(metric)        
        cal_modes = self.calibration_modes
        
        vectors = self._vectors()    
        loss = 0.0
        total_weights = 0.0
        
        for k,(actual,estimated) in vectors.items():
            weight = WEIGHTS[k]  # weight of the objective
            mode_weight = np.ones((len(cal_modes),1))# weight of the modes
            if "distance" in k or "vot" in k:
                mode_weight = self._get_mode_weights(k)  
            
            sel_mode = ([cal_modes.index(i) for i in SELECTED_MODE[k]] 
                        if k in SELECTED_MODE else None)
                   
            x = (actual[sel_mode]*mode_weight[sel_mode]).flatten()
            y = (estimated[sel_mode]*mode_weight[sel_mode]).flatten()
            k_loss = loss_func(x, y) #DO NOT APPLY LOG HERE
            
            if k=="distance":
                # Add the loss of the first bin, it controls better the alphas
                xd0 = (actual[:,:1]*mode_weight)[sel_mode].flatten()
                yd0 = (estimated[:,:1]*mode_weight)[sel_mode].flatten()
                k_loss += 2.5*loss_func(xd0, yd0)
                
            self.losses_record[k].append(k_loss)
            loss += weight*k_loss
            total_weights += weight
              
        return np.log(loss/(10*total_weights)) # I use the log because it highlights better the minima

    

    def plot(self, path:str= None):
        x = self.losses_record
        # stds, avgs, q75 = {}, {}, {}
        for k,y in x.items():
            weight = WEIGHTS[k] 
            y = np.array(y)                                   
            plt.scatter(range(len(y)),y*weight, label=k, alpha=0.5, s=4)
            # print(k, '|STD  -->', np.std(y*weight))
            # print(k, '|Mean -->', np.mean(y*weight))
            # print(k, '|LAST -->', (y*weight)[-1], "\n")            
            # std, mean, q75i = y.std(), y.mean(), np.percentile(y, 75)
            # stds[k] = (float(std))
            # avgs[k] = (float(mean))
            # q75[k]  = (q75i)
        
        plt.ylim([-0.2,5])                    
        plt.legend()
        plt.grid(alpha=0.5)
        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
    
    
    
    
    



# for k,(actual,estimated) in vectors.items():
#     print(k)
#     weight = WEIGHTS[k]   
    
#     for mode in cal_modes:
#         sel_mode = [cal_modes.index(mode)]           
#         x = (actual[sel_mode]*mode_weights[sel_mode]).flatten()
#         y = (estimated[sel_mode]*mode_weights[sel_mode]).flatten()
#         k_loss = loss_func(x, y) #DO NOT APPLY LOG HERE
#         print("   %s --> %.6f"%(mode, k_loss))
    


    