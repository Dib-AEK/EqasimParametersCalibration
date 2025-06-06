#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:46:14 2025

@author: dabdelkader
"""

from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import numpy as np
import polars as pl
from Utilities.Parameters import Parameters

class MetaCls(ABCMeta):
    """
    Metaclass that:
    - For attributes containing certain prefixes, proxies to cls.parameters
    - Otherwise, falls back to normal class-level lookup
    """
    def __getattr__(cls, name):
        _prefixes = ["walk", "bike", "cost", "pt", "car", "swissBike", "swissCar"]
        
        if any(prefix in name for prefix in _prefixes):            
            if hasattr(cls.parameters, name):
                return getattr(cls.parameters, name)
            else:
                raise AttributeError(f"'{name}' not found in {cls.__name__}.parameters" )
        try:
            return object.__getattribute__(cls, name)
        except AttributeError:
            raise AttributeError( f"'{cls.__name__}' object has no attribute '{name}'" )
    
class BaseUtility(ABC, metaclass=MetaCls):
    """
    Abstract base class for utility computation.
    Provides shared structure and namespaced parameters for different transport modes.
    """
    parameters = Parameters
    
    @staticmethod
    def to_yaml(file_path: str):
        BaseUtility.parameters.to_yaml(file_path)
    
    @staticmethod
    def from_yaml(file_path: str):
        BaseUtility.parameters.from_yaml(file_path)
    
    @staticmethod
    def get_parameters(parameters_names: list):
        return BaseUtility.parameters.get_parameters(parameters_names)
        
    @staticmethod
    def set_parameters(updates: dict):
        BaseUtility.parameters.set_parameters(updates)
    


    
    @staticmethod
    def interaction(euclidean_distance_km):
        """
        Computes the distance interaction factor.
        
        Handles both Polars Series and scalar-like inputs.
        """
        lambda_val = BaseUtility.cost.lambdaCostEuclideanDistance
        ref_dist_km = BaseUtility.cost.referenceEuclideanDistance_km
    
        # Case 1: Input is a Polars Series
        if isinstance(euclidean_distance_km, pl.Series):
            euc_distance = euclidean_distance_km.clip(lower_bound=1e-3)
        else:
            euc_distance = np.maximum(euclidean_distance_km, 1e-3)
        
        return (euc_distance / ref_dist_km) ** lambda_val
    
    @staticmethod
    @abstractmethod
    def compute(variables):
        """
        Abstract method to compute utility from variables.
        Must be implemented by subclasses.
        """
        pass

    @staticmethod    
    def read_csv(file_path):
        df = pl.read_csv(file_path, separator=";")
        df = df.with_columns(
            (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trip_index").cast(pl.Utf8)).alias("trip_key")
        )
        return df


