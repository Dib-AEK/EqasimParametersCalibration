#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:55:47 2025

@author: dabdelkader
"""
from Utilities.BaseUtility import BaseUtility
import pandas as pd


class WalkUtility(BaseUtility):
    
    @staticmethod
    def compute(variables):
        utility = (
            BaseUtility.walk.alpha_u +
            BaseUtility.walk.betaTravelTime_u_min * variables["travelTime_min"]
        )
        return utility
    
    def read_csv(file_path):
        df = pd.read_csv(file_path, sep=";")
        df["index"] = df[["person_id", "trip_index"]].apply(lambda x: f"{x['person_id']}_{x['trip_index']}", axis=1)
        df.set_index("index", inplace=True)
        return df
    
    