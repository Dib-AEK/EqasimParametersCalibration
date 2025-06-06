#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:12:48 2025

@author: dabdelkader
"""

from Utilities.BaseUtility import BaseUtility
import pandas as pd
import numpy as np
import polars as pl

class ZeroUtility(BaseUtility):
    
    @staticmethod
    def compute(variables):
        return variables["euclideanDistance_km"]*0.0 #just to make it same ttype
    
    def read_csv(file_path):
        df = pl.read_csv(file_path, separator=";")
        df = df.with_columns(
            (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trip_index").cast(pl.Utf8)).alias("trip_key")
        )
        return df