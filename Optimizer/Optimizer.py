#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:51:39 2025

@author: dabdelkader
"""

import numpy as np
from Loss.Loss import Loss
from Utilities.BaseUtility import BaseUtility
from abc import ABC, abstractmethod

from typing import Dict, Any, Callable
from abc import ABC
from functools import partial



class Optimizer(ABC):
    def __init__(self, args, objective_function: Loss):
        """
        Base class for all optimizers.

        Parameters:
        - objective_function: The black-box function to optimize.
        - bounds: Dictionary of parameter names and their +-deviation from initial values.
        - max_evals: Maximum number of function evaluations.
        """
        self.objective_function = objective_function
        self.bounds = args.bounds
        self.max_evals = args.max_evals
        self.cache_file = ".cache/optimizer_progresssion.p"
        self.args = args
        
        # Get initial values from BaseUtility
        self.param_names = list(self.bounds.keys())
        self.initial_values = BaseUtility.get_parameters(self.param_names)        
        # Convert percentage bounds to absolute bounds
        self.lb = [self.initial_values[p] - self.bounds[p] for p in self.param_names]
        self.ub = [self.initial_values[p] + self.bounds[p] for p in self.param_names]
        
        self.explored_solutions = []
        self.explored_objectives = []

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization and return the best result.

        Returns:
        - Dict containing 'params' and 'loss'
        """
        pass

    def _objective(self, parameters: list, scaled_params:list=None) -> float:
        """Objective function wrapper"""        
        param_dict = dict(zip(self.param_names, parameters))
        loss = float(self.objective_function.get_loss(param_dict))
        
        if scaled_params is None:
            self.explored_solutions.append(list(parameters))
        else:
            self.explored_solutions.append(list(scaled_params))
            
        self.explored_objectives.append(loss)
        return loss

    def get_actual_mode_shares(self, modes=None):
        return self.objective_function.get_actual_mode_shares(modes)

    def get_estimated_mode_shares(self, modes=None):
        return self.objective_function.get_estimated_mode_shares(modes)
    
    def get_eqasim_mode_shares(self, modes=None):
        return self.objective_function.get_eqasim_mode_shares(modes)
    
    def get_current_parameters(self, params):
        if len(params):
            return BaseUtility.get_parameters(params)
        else:
            return []















