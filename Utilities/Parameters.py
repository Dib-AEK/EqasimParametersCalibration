#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:00:03 2025

@author: dabdelkader
"""

import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Parameters(ABC):
    """
    Abstract base class for utility computation.
    Provides shared structure and namespaced parameters for different transport modes.
    """

    class bike:
        alpha_u: float = 0.344
        betaTravelTime_u_min: float = -0.09
        betaAgeOver18_u_a: float = -0.049

    class swissBike:
        betaStatedPreferenceRegion3_u: float = -0.366

    class car:
        alpha_u: float = 0.827
        betaTravelTime_u_min: float = -0.067
        additionalAccessEgressWalkTime_min: float = 4.0
        constantParkingSearchPenalty_min: float = 4.0

    class swissCar:
        betaStatedPreferenceRegion1_u: float = -0.4
        betaStatedPreferenceRegion3_u: float = 0.4

    class walk:
        alpha_u: float = 1.3
        betaTravelTime_u_min: float = -0.141

    class pt:
        alpha_u: float = 0.0
        betaLineSwitch_u: float = -0.17
        betaInVehicleTime_u_min: float = -0.019
        betaWaitingTime_u_min: float = -0.038
        betaAccessEgressTime_u_min: float = -0.08

    class cost:        
        betaCost_u_MU: float = -0.126
        lambdaCostEuclideanDistance: float = -0.4
        referenceEuclideanDistance_km: float = 40.0

    @classmethod
    def to_yaml(cls, file_path: str):        
        """
        Save all parameters to a YAML file.
        """
        data = {}

        for name, obj in inspect.getmembers(cls):
            if inspect.isclass(obj) and obj != cls:                
                for key, value in inspect.getmembers(obj):
                    if (not key.startswith('__')) and (isinstance(value, float)):
                        if name=="cost":
                            data[key] = float(value)
                        else:                        
                            data[name+"."+key] = float(value)

        with open(file_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        
        logger.info(f"All DMC parameters saved to: {file_path}")

    @classmethod
    def from_yaml(cls, file_path: str):
        """
        Load parameters from a YAML file, updating class attributes.
        """
        logger.info(f"Reading DMC parameters from: {file_path}")
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        for k, v in data.items():
            if "." in k:
                class_name, attribute_name = k.split('.')
            else:
                class_name, attribute_name = "cost", k
            
            if hasattr(cls, class_name):
                section_class = getattr(cls, class_name)
                if hasattr(section_class, attribute_name):
                    setattr(section_class, attribute_name, v)
                        
    @staticmethod
    def get_parameters(parameters_names: list):
        """
        Retrieve the parameter classes for the given names.
        """
        if len(parameters_names):
            out_dict = dict()
            for name in parameters_names:
                class_name, attr_name = name.split('.', 1)
                param_class = getattr(Parameters, class_name, None)
                if param_class is None:
                    raise ValueError(f"Unknown parameter: '{class_name}'")
                    
                out_dict[name] = getattr(param_class, attr_name, None)
                if out_dict[name] is None:
                    raise ValueError(f"Unknown parameter: '{name}'")
                
            return out_dict 

    @staticmethod
    def set_parameters(updates: dict):
        """
        Update parameters using dot notation keys like 'car.alpha_u'.
    
        Parameters:
        - updates: dict of form {
            'car.alpha_u': 0.9,
            'pt.betaWaitingTime_u_min': -0.04
          }
        """
        if len(updates):
            for key, value in updates.items():
                try:
                    class_name, attr_name = key.split('.', 1)
                except ValueError:
                    raise ValueError(f"Invalid key format: '{key}', expected 'class_name.attribute_name'")
                
                param_class = getattr(Parameters, class_name, None)
                if param_class is None:
                    raise ValueError(f"Unknown parameter group: '{class_name}'")
                
                if not hasattr(param_class, attr_name):
                    raise AttributeError(f"'{class_name}' has no attribute '{attr_name}'")
                
                setattr(param_class, attr_name, value)
