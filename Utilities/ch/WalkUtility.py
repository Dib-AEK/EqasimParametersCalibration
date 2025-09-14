#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:55:47 2025

@author: dabdelkader
"""
from .BaseUtility import BaseUtility
import pandas as pd
import polars as pl

class WalkUtility(BaseUtility):
    
    @staticmethod
    def compute(variables):
        utility = (
            BaseUtility.walk.alpha_u +
            BaseUtility.walk.betaTravelTime_u_min * variables["travelTime_min"]
        )
        return utility
    
    @staticmethod
    def compute_lazy():
        utility = (
            BaseUtility.walk.alpha_u +
            BaseUtility.walk.betaTravelTime_u_min * pl.col("travelTime_min")
        )
        return utility
    
    def read_csv(file_path):
        df = pl.read_csv(file_path, separator=";")
        df = df.with_columns(
            (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trip_index").cast(pl.Utf8)).alias("trip_key")
        )
        return df
    
    