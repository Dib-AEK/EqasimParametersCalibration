#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 18:16:46 2025

@author: dabdelkader
"""
from Utilities.CarUtility import CarUtility
from Utilities.BaseUtility import BaseUtility
import pandas as pd
import polars as pl

class PtUtility(BaseUtility):
    
    @staticmethod
    def compute(variables):
        """
        Computes the total utility for public transport mode.
    
        Parameters:
        - variables (dict): Should include keys:
            - "accessEgressTime_min", "inVehicleTime_min", "waitingTime_min",
              "numberOfLineSwitches", "cost_MU", "euclideanDistance_km"
    
        Returns:
        - float: The computed utility value.
        """
        utility = 0.0
    
        utility += BaseUtility.pt.alpha_u
        utility += BaseUtility.pt.betaAccessEgressTime_u_min * variables["accessEgressTime_min"]
        utility += BaseUtility.pt.betaInVehicleTime_u_min * variables["inVehicleTime_min"]
        utility += BaseUtility.pt.betaWaitingTime_u_min * variables["waitingTime_min"]
        utility += BaseUtility.pt.betaLineSwitch_u * variables["numberOfLineSwitches"]
    
        cost_interaction = BaseUtility.interaction(variables["euclideanDistance_km"])
        utility += BaseUtility.cost.betaCost_u_MU * cost_interaction * variables["cost_MU"]
        return utility
    
    @staticmethod
    def compute_lazy():
        utility = 0.0
    
        utility += BaseUtility.pt.alpha_u
        utility += BaseUtility.pt.betaAccessEgressTime_u_min * pl.col("accessEgressTime_min")
        utility += BaseUtility.pt.betaInVehicleTime_u_min * pl.col("inVehicleTime_min")
        utility += BaseUtility.pt.betaWaitingTime_u_min * pl.col("waitingTime_min")
        utility += BaseUtility.pt.betaLineSwitch_u * pl.col("numberOfLineSwitches")
    
        cost_interaction = BaseUtility.interaction_lazy(pl.col("euclideanDistance_km"))
        utility += BaseUtility.cost.betaCost_u_MU * cost_interaction * pl.col("cost_MU")
        return utility
    
    def read_csv(file_path):
        df = pl.read_csv(file_path, separator=";")
        df = df.with_columns(
            (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trip_index").cast(pl.Utf8)).alias("trip_key")
        )
        return df
    
    
    
    
    
    
    
    
    
    
    
    
    