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
        self.cache_file = os.path.join(args.optimizer_cache, "optimizer_progression.p")
        self.cache_solution_file = os.path.join(args.optimizer_cache, "optimizer_explored_solutions.json")
        self.args = args

        self.image_path = os.path.join(args.optimizer_cache, f"{args.iteration}.optimizer_progression.png")
        # Get initial values from BaseUtility
        self.param_names = list(self.bounds.keys())
        self.initial_values = BaseUtility.get_parameters(self.param_names)

        # Convert relative bounds to absolute bounds
        self.lb = [self.initial_values[p] - self.bounds[p] for p in self.param_names]
        self.ub = [self.initial_values[p] + self.bounds[p] for p in self.param_names]
        
        self.explored_solutions = []
        self.explored_objectives = []

    @abstractmethod
    def optimize(self, *args, **kwargs) -> Dict[str, Any]:
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

    def save_explored_solutions_and_objectives(self):
        data = dict(explored_solutions = self.explored_solutions,
                    explored_objectives = self.explored_objectives)
        file_path = self.cache_solution_file
        # save it as json file
        with open(file_path, "w") as f:
            json.dump(data, f)

    def load_explored_solutions_and_objectives(self):
        file_path = self.cache_solution_file
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            self.explored_solutions = data.get("explored_solutions", [])
            self.explored_objectives = data.get("explored_objectives", [])

    def plot(self, show = False):
        l = self.explored_solutions
        if len(l):
            l = [list(li) for li in l]
            l = np.array(l)
            o = self.explored_objectives
            
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