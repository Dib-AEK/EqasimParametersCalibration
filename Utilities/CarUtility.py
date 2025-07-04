#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:48:11 2025

@author: dabdelkader
"""
from Utilities.BaseUtility import BaseUtility
import pandas as pd
import numpy as np
import polars as pl

class CarUtility(BaseUtility):    
    
    @staticmethod
    def estimateRegionalUtility(variables):
        beta1 = BaseUtility.swissCar.betaStatedPreferenceRegion1_u
        beta3 = BaseUtility.swissCar.betaStatedPreferenceRegion3_u
        
        if isinstance(variables, pl.DataFrame):
            # Use Polars expressions for efficient conditional logic
            return ( variables.select(
                    pl.when(pl.col("statedPreferenceRegion") == 1)
                    .then(beta1)
                    .when(pl.col("statedPreferenceRegion") == 3)
                    .then(beta3)
                    .otherwise(0.0)
                    .alias("regional_utility")
                    ).to_series())
        
        elif isinstance(variables, dict) or (isinstance(variables, pd.Series) and "statedPreferenceRegion" in variables and variables.ndim == 1):
            # Handles dict or row Series
            region = variables["statedPreferenceRegion"]
            if region == 1:
                return beta1
            elif region == 3:
                return beta3
            else:
                return 0.0
    
        elif isinstance(variables, pd.DataFrame):
            region = variables["statedPreferenceRegion"]
            return np.select([region == 1, region == 3], [beta1, beta3], default=0.0)
    
        elif isinstance(variables, pd.Series):
            # Assume this is a Series of region values
            return variables.map(lambda x: beta1 if x == 1 else beta3 if x == 3 else 0.0)
    
        else:
            raise TypeError("Unsupported input type.")

		    
    @staticmethod
    def compute(variables):
        """
        Computes the total utility for car mode.

        Parameters:
        - variables (dict): Should include keys:
            - "travelTime_min", "accessEgressTime_min", "cost_MU", "euclideanDistance_km"

        Returns:
        - float: The computed utility value.
        """
        utility = 0.0

        utility += BaseUtility.car.alpha_u
        utility += BaseUtility.car.betaTravelTime_u_min * variables["travelTime_min"]
        utility += BaseUtility.walk.betaTravelTime_u_min * variables["accessEgressTime_min"]

        cost_interaction = BaseUtility.interaction( variables["euclideanDistance_km"] )
        utility += BaseUtility.cost.betaCost_u_MU * cost_interaction * variables["cost_MU"]
        utility += CarUtility.estimateRegionalUtility(variables)
        
        return utility
    
    @staticmethod
    def compute_lazy():
        utility = BaseUtility.car.alpha_u
    
        utility += BaseUtility.car.betaTravelTime_u_min * pl.col("travelTime_min")
        utility += BaseUtility.walk.betaTravelTime_u_min * pl.col("accessEgressTime_min")
        
        cost_interaction = BaseUtility.interaction_lazy( pl.col("euclideanDistance_km") )
        utility += BaseUtility.cost.betaCost_u_MU * cost_interaction * pl.col("cost_MU")
    
        # Region-specific utility using lazy expression
        beta1 = BaseUtility.swissCar.betaStatedPreferenceRegion1_u
        beta3 = BaseUtility.swissCar.betaStatedPreferenceRegion3_u
    
        regional_utility = (
            pl.when(pl.col("statedPreferenceRegion") == 1)
            .then(beta1)
            .when(pl.col("statedPreferenceRegion") == 3)
            .then(beta3)
            .otherwise(0.0)
        )
    
        utility += regional_utility
    
        return utility

    
    def read_csv(file_path):
        df = pl.read_csv(file_path, separator=";")
        df = df.with_columns(
            (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trip_index").cast(pl.Utf8)).alias("trip_key")
        )
        return df
        
        