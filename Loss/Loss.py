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
import time
import polars as pl

class Loss:
    
    def __init__(self, mode_shares_provider: ModeShares,
                       metric = "js", 
                       calibrate_global_modeshare:bool=True, 
                       calibrate_modeshare_distribution:bool=False,
                       distributions:list = None):
        """
        Parameters:
        - tours: pd.DataFrame with a 'chosen_mode' column (representing selected transport modes)
        - actual_mode_shares: dict, e.g. {'car': 0.4, 'pt': 0.3, 'walk': 0.2, 'bike': 0.1}
        - mode_shares_distribution: dict,similar to actual_mode_shares but with a list 
        for each mode and an extra key (distance)
        """        
        self.actual_global_mode_shares = mode_shares_provider.get_mode_shares()
        self.actual_mode_shares_distribution = mode_shares_provider.get_mode_shares_distribution()
        # get the actual distributions by some attribute
        for attrb in ["age","income","canton"]:            
            value = getattr(mode_shares_provider, f"mode_shares_by_{attrb}")
            setattr(self, f"actual_mode_shares_{attrb}", value)
        
        # get actual distance distribution of modes
        for mode in ["car","pt","walk","bike"]:            
            value = getattr(mode_shares_provider, f"{mode}_distance_distribution")
            setattr(self, f"{mode}_distance", value)
        
        
        self.distance_bins = mode_shares_provider.get_distance_bins()
        self.distance_labels = mode_shares_provider.get_distance_labels()
        
        self.calibrate_global_modeshare = calibrate_global_modeshare
        self.calibrate_modeshare_distribution = calibrate_modeshare_distribution
        self.metric = metric
        
        self.distributions = distributions
        if distributions is not None:
            list_of_dist_groups = ["distance","canton","age","income"]
            list_of_dist_groups.extend([f"{mode}_distance" for mode in ["car","pt","walk","bike"]])            
            assert all([d in list_of_dist_groups for d in distributions])
        
        self.modes = ["car","walk","bike","pt","car_passenger"]         
        self.calibration_modes = ["car","pt","bike","walk"]
        # self.utility_time = []
        # self.selector_time = []
        # self.mode_share_time = []

    def get_loss(self, parameters=None):
        if parameters is not None:
            BaseUtility.set_parameters(parameters)
        return self._get_loss()
    
    
    @property
    def estimated_mode_shares(self):
        return self.get_estimated_mode_shares()
    
    def get_estimated_mode_shares(self, modes=None):
        """
        Computes estimated mode shares from the tours.
        
        Returns:
            tuple:
                - dict of global estimated mode shares
                - dict of estimated shares by distance, age, income, canton
        """
        # Use default modes if not provided
        if modes is None:
            modes = self.modes
    
        # Retrieve and filter tours
        tours = TourUtility.get_all_utilities().collect()
        tours = Selector.select(tours)
    
        # Select and explode relevant columns
        base_columns = ["candidate_mode", "euclidean_distance", "income_class", "canton_id", "age_class"]
        exploded_columns = ["candidate_mode", "euclidean_distance"]
    
        selected_modes = tours.select(base_columns).explode(exploded_columns).filter(pl.col("euclidean_distance") > 1e-3)
        
        # Initialize result containers
        global_mode_share = {}
        mode_share_by_group = {}
    
        # Global mode share estimation
        if self.calibrate_global_modeshare:
            global_mode_share = self._estimate_global_mode_share(selected_modes, modes)
    
        # Distributional mode share estimation
        considered_distributions = self.distributions
        if considered_distributions is None:
            considered_distributions = ["distance","age","income","canton"]
            
        if self.calibrate_modeshare_distribution:
            if "distance" in considered_distributions:
                mode_share_by_group["distance"] = self._estimate_mode_share_distribution_distance(selected_modes, modes)
            if "age" in considered_distributions:    
                mode_share_by_group["age"]      = self._estimate_mode_share_distribution_by(selected_modes, modes, "age_class")
            if "income" in considered_distributions:
                mode_share_by_group["income"]   = self._estimate_mode_share_distribution_by(selected_modes, modes, "income_class")
            if "canton" in considered_distributions:
                mode_share_by_group["canton"]   = self._estimate_mode_share_distribution_by(selected_modes, modes, "canton_id")
            
            for mode in ["car","pt","bike","walk"]:
                if f"{mode}_distance" in considered_distributions:
                    mode_share_by_group[f"{mode}_distance"]   = self._estimate_mode_distance_distribution(selected_modes, mode)    
            
            
        return global_mode_share, mode_share_by_group



    def _estimate_global_mode_share(self, selected_modes, modes):
        estimates_global_mode_share = dict()
        counts_df = selected_modes.group_by("candidate_mode").agg(pl.count().alias("count"))
        total = counts_df["count"].sum()
        counts_df = counts_df.with_columns( (pl.col("count") / total).alias("share") )
        counts_dict = dict(zip(counts_df["candidate_mode"], counts_df["share"]))
        estimates_global_mode_share = {mode: counts_dict.get(mode, 0.0) for mode in modes}
        return estimates_global_mode_share
    
    def _estimate_mode_share_distribution_distance(self, selected_modes, modes):
        distance_bins = np.array(self.distance_bins)*1e-3 #convert to km
        bin_labels    =  self.distance_labels
        selected_modes = selected_modes.with_columns([
            pl.col("euclidean_distance").cut(breaks=distance_bins[1:-1], 
                                             labels=bin_labels).alias("distance_bin")])
        
        counts =  selected_modes.group_by(["distance_bin", "candidate_mode"]).agg(pl.count().alias("count"))
        pivoted = counts.pivot(values="count", index="distance_bin", columns="candidate_mode", aggregate_function="first").fill_null(0)
        pivoted = pivoted.sort("distance_bin")
        
        for mode in modes:
            if mode not in pivoted.columns:
                pivoted = pivoted.with_columns(pl.lit(0).alias(mode))
                
        mode_shares_by_bin = pivoted .with_columns([
                (pl.col(col) / pl.sum_horizontal(pl.exclude("distance_bin"))).alias(col)
                for col in pivoted.columns if col != "distance_bin" ])
        
        estimates_mode_share_distribution = {
                mode: mode_shares_by_bin.select(mode).to_series().to_list()
                for mode in modes
            }
        return estimates_mode_share_distribution
    
    def _estimate_mode_share_distribution_by(self, selected_modes, modes, by = "canton_id"):
               
        pivoted = (selected_modes.select([by, "candidate_mode"])
                                 .group_by([by, "candidate_mode"])
                                 .agg(pl.len().alias("count"))
                                 .pivot(values="count", index=by, on="candidate_mode", aggregate_function=None)
                                 .sort(by)
                                 .fill_null(0)                                 
                                 .with_columns(
                                         pl.exclude(by)/ pl.sum_horizontal(pl.exclude(by))
                                         )
                                 )                
        
        for mode in modes:
            if mode not in pivoted.columns:
                pivoted = pivoted.with_columns(pl.lit(0).alias(mode))
                
        estimates_mode_share_distribution = {
                mode: pivoted.select(mode).to_series().to_list()
                for mode in modes
            }
        return estimates_mode_share_distribution
    
    
    def _estimate_mode_distance_distribution(self, selected_modes, mode):
        distance_bins = np.array(self.distance_bins)*1e-3 #convert to km
        bin_labels    =  self.distance_labels
        
        df = (selected_modes
                        .filter(pl.col("candidate_mode") == mode)
                        .select(["candidate_mode", "euclidean_distance"])
                        .with_columns([
                            pl.col("euclidean_distance")
                              .cut(breaks=distance_bins[1:-1], labels=bin_labels)
                              .alias("distance_bin")
                        ])
                        .group_by("distance_bin")
                        .agg(pl.len().alias("count"))
                        .with_columns((pl.col("count")/pl.col("count").sum()).alias("share"))
                        .select(["distance_bin","share"])
                        .with_columns(pl.col("distance_bin").cast(pl.String))
                    )

        all_bins_df = pl.DataFrame({"distance_bin": bin_labels})
        
        df_complete = (
            all_bins_df
            .join(df, on="distance_bin", how="left")
            .with_columns(pl.col("share").fill_null(0.0))
        )
        return np.array(df_complete["share"].to_list())
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    @property
    def actual_mode_shares(self):
        return self.get_actual_mode_shares()
    
    def get_actual_mode_shares(self, modes=None):
        """
        Returns the actual (observed) mode shares used for calibration.
    
        Args:
            modes (list, optional): List of modes to return shares for. Defaults to self.modes.
    
        Returns:
            tuple:
                - dict of global mode shares
                - dict of mode shares by distance, age, income, and canton
        """
        if modes is None:
            modes = self.modes
    
        # Helper function to fetch mode shares from attribute by mode list
        def get_shares(attr):
            return {mode: attr.get(mode, 0.0) for mode in modes}
    
        # Global and distributional actual shares
        global_shares = get_shares(self.actual_global_mode_shares)
        shares_by_group = {
            "distance": get_shares(self.actual_mode_shares_distribution),
            "age":      get_shares(self.actual_mode_shares_age),
            "income":   get_shares(self.actual_mode_shares_income),
            "canton":   get_shares(self.actual_mode_shares_canton),
            "car_distance": self.car_distance,
            "walk_distance": self.walk_distance,
            "bike_distance": self.bike_distance,
            "pt_distance": self.pt_distance,
        }
    
        return global_shares, shares_by_group

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
        selected_modes = TourUtility.tours.loc[TourUtility.tours["eqasim_selected"], cols].explode(column=cols)
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
        """
        Compute the total calibration loss across global and distributional mode shares.
    
        Returns:
            float: Total loss scaled by 100.0
        """
        metric = self.metric.lower()
        loss_func = self.get_loss_func(metric)
    
        (actual_vec, est_vec), distribution = self._vectors()
    
        loss = 0.0
    
        considered_distributions = self.distributions
        if considered_distributions is None:
            considered_distributions = list(distribution.keys())
            
        # Global mode share loss
        if self.calibrate_global_modeshare and actual_vec is not None and est_vec is not None:
            loss += loss_func(actual_vec, est_vec)
    
        # Distributional mode share loss
        if self.calibrate_modeshare_distribution:
            for key, (actual_group_vec, est_group_vec) in distribution.items():
                if key in considered_distributions:
                    loss += loss_func(actual_group_vec.flatten(),
                                      est_group_vec.flatten())
        
        num_losses = len(considered_distributions)+1
        loss = loss/num_losses
        return np.log(loss)



    def _vectors(self):
        """
        Returns actual and estimated mode shares as aligned NumPy arrays.
    
        Returns:
            tuple:
                - (actual_vec, estimated_vec): Global mode share vectors
                - est_distributions: dict of (actual_vec, estimated_vec) pairs by group (distance, age, etc.)
        """
        # Retrieve mode share data
        est, est_by_group = self.estimated_mode_shares
        actual, actual_by_group = self.actual_mode_shares
        modes = self.calibration_modes
    
        global_actual_vec, global_est_vec = None, None
        distributions = {}
    
        # Global mode share vectors
        if self.calibrate_global_modeshare:
            global_actual_vec = np.array([actual[mode] for mode in modes])
            global_est_vec = np.array([est[mode] for mode in modes])
    
        # Distributional mode share vectors by group
        if self.calibrate_modeshare_distribution:
            for group_key in est_by_group:
                actual_group = actual_by_group[group_key]
                estimated_group = est_by_group[group_key]
                
                if isinstance(estimated_group, dict):
                    actual_vec = np.array([actual_group.get(mode, 0.0) for mode in modes])
                    estimated_vec = np.array([estimated_group.get(mode, 0.0) for mode in modes])
    
                distributions[group_key] = (actual_vec, estimated_vec)
    
        return (global_actual_vec, global_est_vec), distributions
    
    
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
        elif metric=="ll" or metric=="log_likelihood":
            return self.log_likelihood()
        
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

    def log_likelihood(self):
        def func(x,y):
            actual_probs = np.clip(x, 1e-6, 1)
            actual_probs/=actual_probs.sum()
            
            pred_probs = np.clip(y, 1e-6, 1) 
            pred_probs/=pred_probs.sum()
            
            logL = np.sum(actual_probs * np.log(pred_probs))
            return -logL             
            
        return func