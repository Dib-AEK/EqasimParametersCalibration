#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:39:19 2025

@author: dabdelkader
"""

from abc import ABC, abstractmethod
import json
import numpy as np
from typing import Dict, Optional
import os
import MomentumAndDecay.utils as U

class MomentumBase(ABC):
    """
    Abstract base class for momentum-based smoothing of parameter updates.
    """
    def __init__(self, cache_path: str):
        self.cache_path  = os.path.join(cache_path, "momentum_cache.json")
        self.param_keys  = []
        self.hyperparams = {}
        self.state       = {}

        self.smoothed_values = np.array([])
        self.initial_values = np.array([])
        self.optimal_values = np.array([])        
    
    def _save_cache(self) -> None:
        """Save current state to disk."""
        data = dict(param_keys=self.param_keys, 
                    hyperparams=self.hyperparams, 
                    state=self.state)
        # convert any np.array to list
        data = U.to_lists(data)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_cache(self) -> None:
        """Load internal state from disk."""        
        data = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load cache: {e}")
        data = U.to_arrays(data)  # Convert lists back to np.arrays
        
        self.param_keys = data.get("param_keys", [])
        self.hyperparams = data.get("hyperparams", {})
        self.state = data.get("state", {})

    def set_initial_values(self, values: Dict[str, float]) -> None:
        """
        Set initial parameter values (x₀).
        
        Args:
            values: Dictionary of parameter names and their initial values.
        """
        values = values.copy()        
        self.param_keys = list(values.keys())
        self.initial_values = np.array([values[k] for k in self.param_keys], dtype=np.float64)
    
    def set_optimal_values(self, values: Dict[str, float]) -> None:
        """
        Set optimized parameter values (x₁) and update the smoother.
        
        Args:
            values: Dictionary of parameter names and their optimized values.
        """
        values = values.copy()
        if hasattr(self, 'initial_values'):
            # Verify keys match
            if values.keys() != self.get_initial_values().keys():
                raise ValueError("Parameters keys must match between initial and optimized values.")
        
        self.optimal_values = np.array([values[k] for k in self.param_keys], dtype=np.float64)
        self._update_smoother()
    
    def get_updated_values(self) -> Dict[str, float]:
        """
        Get the smoothed/damped parameter values to limit variations.
        
        Returns:
            Dictionary of parameter names and their smoothed values.
        """
        if not self.param_keys:
            raise RuntimeError("No parameters have been set yet.")
            
        return {
            k: float(v) for k, v in zip(self.param_keys, self.smoothed_values)
        }
    
    @abstractmethod
    def _update_smoother(self, optimized_values: Dict[str, float]) -> None:
        """Internal method to update the smoothing state."""
        pass
    
    def get_initial_values(self) -> Dict[str, float]:
        """Convert initial values array back to dictionary."""
        if not hasattr(self, 'initial_values'):
            return {}
        return {k: float(v) for k, v in zip(self.param_keys, self.initial_values)}
    
    def get_optimal_values(self) -> Dict[str, float]:
        """Convert optimal_values values array back to dictionary."""
        if not hasattr(self, 'optimal_values'):
            return {}
        return {k: float(v) for k, v in zip(self.param_keys, self.optimal_values)}
    
    def get_smoothed_values(self) -> Dict[str, float]:
        """Convert smoothed_values values array back to dictionary."""
        if not hasattr(self, 'smoothed_values'):
            return {}
        return {k: float(v) for k, v in zip(self.param_keys, self.smoothed_values)}