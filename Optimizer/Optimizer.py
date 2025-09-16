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
import os
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
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
        optimizer = args.optimizer.lower()
        self.cache_file = os.path.join(args.optimizer_cache, f"{optimizer}_optimizer_progression.p")
        self.cache_state_file = os.path.join(args.optimizer_cache, f"{optimizer}_optimizer_state.json")
        self.args = args
        self.modes_to_calibrate = args.modes_in_loss
        
        self.image_path = os.path.join(args.optimizer_cache, f"{args.iteration}.optimizer_progression.png")
        # Get initial values from BaseUtility
        self.param_names = list(self.bounds.keys())
        self.initial_values = BaseUtility.get_parameters(self.param_names)

        # Convert relative bounds to absolute bounds
        self.lb = [self.initial_values[p] - self.bounds[p] for p in self.param_names]
        self.ub = [self.initial_values[p] + self.bounds[p] for p in self.param_names]
        
        self.set_scalers()
        
        self.explored_solutions = []
        self.explored_scaled_solutions = []
        self.explored_objectives = []
    
    def set_scalers(self):
        assert len(self.lb)>0, "Bounds cannot be empty."
        assert len(self.lb) == len(self.ub), "Lower and upper bounds must have the same length."
        assert all(ub > lb for lb, ub in zip(self.lb, self.ub)), "Each upper bound must be greater than the corresponding lower bound."
        
        lb, ub = np.array(self.lb), np.array(self.ub)
        self.params_scaler = lambda x: (x - lb) / (ub - lb)
        self.params_unscaler = lambda x: x * (ub - lb) + lb

    @abstractmethod
    def _optimize(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Internal method to run the optimization algorithm.

        Returns:
        - Dict containing 'params' and 'loss'
        """
        pass

    def optimize(self, *args, overwrite=False, **kwargs) -> Dict[str, Any]:
        """
        Run the optimization and return the best result.

        Parameters:
        - overwrite: If True, load the optimizer state from cache before starting.

        Returns:
        - Dict containing 'params' and 'loss'
        """
        if not overwrite:
            self.load_state()

        results = self._optimize(*args, overwrite=overwrite, **kwargs)
        self.save_state()
        return results

    def _objective(self, scaled_parameters: list) -> float:
        """Objective function wrapper"""   
        parameters = self.params_unscaler(scaled_parameters)

        param_dict = dict(zip(self.param_names, parameters))
        loss = float(self.objective_function.get_loss(param_dict))

        self.explored_solutions.append(parameters.tolist())
        self.explored_scaled_solutions.append(scaled_parameters.tolist())
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

    def save_state(self):
        data = dict(explored_solutions = self.explored_solutions,
                    explored_objectives = self.explored_objectives,
                    explored_scaled_solutions = self.explored_scaled_solutions,
                    lb = self.lb,
                    ub = self.ub,
                    param_names = self.param_names,
                    initial_values = self.initial_values,
                    )
        file_path = self.cache_state_file
        # save it as json file
        with open(file_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved optimization state to cache with {len(self.explored_solutions)} evaluated solutions.")

    def load_state(self):
        file_path = self.cache_state_file
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            param_names = data.get("param_names", [])
            if (param_names==self.param_names):
                self.lb = data["lb"]
                self.ub = data["ub"]
                self.explored_solutions = data["explored_solutions"]
                self.explored_objectives = data["explored_objectives"]
                self.explored_scaled_solutions = data["explored_scaled_solutions"]
                self.initial_values = data["initial_values"]
                self.set_scalers()
                logger.info(f"Loaded optimization state from cache with {len(self.explored_solutions)} evaluated solutions.")
            else:
                logger.warning("Optimization state found in the cache, but the cached state does not match current parameters.")
        else:
            logger.info("No optimization state found in the cache. Starting fresh.")

    def plot(self, show = False, max_len=6000):
        l = self.explored_scaled_solutions[-max_len:]  # get the last max_len solutions
        if len(l):
            l = [list(li) for li in l]
            l = np.array(l)
            o = self.explored_objectives[-max_len:]  # get the last max_len objectives

            fig, ax = plt.subplots(2,1,figsize=(12,8))
            ax[0].grid(alpha=0.3)
            ax[1].grid(alpha=0.3)
            
            for i,param in enumerate(self.param_names):
                y = l[:,i]
                x = range(len(y))
                ax[0].scatter(x,y, label = param, s=10)
                
            ax[1].scatter(range(len(o)), o, s=10)
                    
            low, high = sorted([min(o), max(o)])        
            ax[1].text(len(o) * 0.8, (high+low)/2, f"Min: {min(o):.3f}", fontsize=15)
            
            ax[0].legend(ncols=int(np.ceil(l.shape[1]/2)), loc='upper center',bbox_to_anchor=(0.5, 1.2),  frameon=False)
    
            if self.image_path is not None:
                plt.savefig(self.image_path, bbox_inches='tight', dpi=300)
            
            if show:
                plt.show()
            else:
                plt.close()