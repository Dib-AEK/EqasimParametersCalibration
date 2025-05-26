#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 18:16:46 2025

@author: dabdelkader
"""
from Utilities.CarUtility import CarUtility
from Utilities.BaseUtility import BaseUtility
import pandas as pd


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
    
    
    def read_csv(file_path):
        df = pd.read_csv(file_path, sep=";")
        df["index"] = df[["person_id", "trip_index"]].apply(lambda x: f"{x['person_id']}_{x['trip_index']}", axis=1)
        df.set_index("index", inplace=True)
        return df
    
    
    
    
    
    
    
    
    
    
    
    
    