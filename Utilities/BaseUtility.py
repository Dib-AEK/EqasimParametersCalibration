#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:46:14 2025

@author: dabdelkader
"""

from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import numpy as np
from Utilities.Parameters import Parameters

class MetaCls(ABCMeta):
    """
    Metaclass that proxies unknown class-level attribute lookups to BaseUtility.parameters.
    """
    def __getattr__(cls, name):
        if ("walk" in name) or ("bike" in name) or ("cost" in name) or("pt" in name) or ("car" in name):
            return getattr(cls.parameters, name)
        else:
            return getattr(cls, name)
    
    
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
    def interaction(euclidean_distance_km: float) -> float:
        """
        Computes the distance interaction factor.
        """
        euc_distance = np.maximum(euclidean_distance_km, 1e-3)
        lambda_val   = BaseUtility.cost.lambdaCostEuclideanDistance
        reference_distance_km = BaseUtility.cost.referenceEuclideanDistance_km
        return (euc_distance / reference_distance_km) ** lambda_val
    
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
        """
        Abstract method to read the csv file created by MATSim.
        Must be implemented by subclasses.
        """
        df = pd.read_csv(file_path, sep=";")
        df["index"] = df[["person_id", "trip_index"]].apply(lambda x: f"{x['person_id']}_{x['trip_index']}", axis=1)
        df.set_index("index", inplace=True)
        return df


