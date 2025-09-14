#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:21:26 2025

@author: dabdelkader
"""
from .BaseUtility import BaseUtility
import pandas as pd
import numpy as np
import polars as pl

class BikeUtility(BaseUtility):
    
    @staticmethod
    def estimateRegionalUtility(variables):
        #(BaseUtility.bike.betaStatedPreferenceRegion3_u if variables["statedPreferenceRegion"] == 3 else 0.0)
        beta1 = 0.0
        beta3 = BaseUtility.swissBike.betaStatedPreferenceRegion3_u
        
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
    def estimateAgeUtility(variables):
        beta = BaseUtility.bike.betaAgeOver18_u_a
    
        if isinstance(variables, pl.DataFrame):
            expr = beta * pl.max_horizontal(0.0, pl.col("age_a") - 18)
            return variables.select(expr.alias("utility"))["utility"]
        else:
            return beta * np.maximum(0.0, variables["age_a"] - 18)
    
        
                
            
    @staticmethod
    def compute(variables):
        """
        Computes the utility based on the provided input data.

        Parameters:
        - variables (dict): A dictionary containing the keys:
            - "travelTime_min": float
            - "age_a": float
            - "statedPreferenceRegion": int

        Returns:
        - float: The computed utility value.
        """
        utility = (
            BaseUtility.bike.alpha_u +
            BaseUtility.bike.betaTravelTime_u_min * variables["travelTime_min"] +
            BikeUtility.estimateAgeUtility(variables)+
            BikeUtility.estimateRegionalUtility(variables)            
        )
        return utility

    @staticmethod
    def compute_lazy():
        betaAge = BaseUtility.bike.betaAgeOver18_u_a
        ageUtility = betaAge * pl.max_horizontal(0.0, pl.col("age_a") - 18)
        
        beta1 = 0.0
        beta3 = BaseUtility.swissBike.betaStatedPreferenceRegion3_u
        regionalUtility = (
            pl.when(pl.col("statedPreferenceRegion") == 1)
            .then(beta1)
            .when(pl.col("statedPreferenceRegion") == 3)
            .then(beta3)
            .otherwise(0.0)
        )
        
        utility = (
            BaseUtility.bike.alpha_u +
            BaseUtility.bike.betaTravelTime_u_min * pl.col("travelTime_min") +
            ageUtility+
            regionalUtility            
        )
        return utility
    
    def read_csv(file_path):
        df = pl.read_csv(file_path, separator=";")
        df = df.with_columns(
            (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trip_index").cast(pl.Utf8)).alias("trip_key")
        )
        return df
    
    
    
    
    
    
    
    
    
    