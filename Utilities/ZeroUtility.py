#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:12:48 2025

@author: dabdelkader
"""

from Utilities.BaseUtility import BaseUtility
import pandas as pd

class ZeroUtility(BaseUtility):
    
    @staticmethod
    def compute(variables):
        return 0.0
    
    def read_csv(file_path):
        df = pd.read_csv(file_path, sep=";")
        df["index"] = df[["person_id", "trip_index"]].apply(lambda x: f"{x['person_id']}_{x['trip_index']}", axis=1)
        df.set_index("index", inplace=True)
        return df