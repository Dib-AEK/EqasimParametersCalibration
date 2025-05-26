#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:21:26 2025

@author: dabdelkader
"""
from Utilities.BaseUtility import BaseUtility
import pandas as pd
import numpy as np

class BikeUtility(BaseUtility):
    
    @staticmethod
    def estimateRegionalUtility(variables):
        #(BaseUtility.bike.betaStatedPreferenceRegion3_u if variables["statedPreferenceRegion"] == 3 else 0.0)
        beta1 = 0.0
        beta3 = BaseUtility.swissBike.betaStatedPreferenceRegion3_u
    
        if isinstance(variables, dict) or (isinstance(variables, pd.Series) and "statedPreferenceRegion" in variables and variables.ndim == 1):
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
            BaseUtility.bike.betaAgeOver18_u_a * np.maximum(0.0, variables["age_a"] - 18) +
            BikeUtility.estimateRegionalUtility(variables)            
        )
        return utility

        
    
    def read_csv(file_path):
        df = pd.read_csv(file_path, sep=";")
        df["index"] = df[["person_id", "trip_index"]].apply(lambda x: f"{x['person_id']}_{x['trip_index']}", axis=1)
        df.set_index("index", inplace=True)
        return df
    
    
    
    
    
    
    
    
    
    