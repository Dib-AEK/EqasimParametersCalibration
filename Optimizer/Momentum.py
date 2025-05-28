#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:39:19 2025

@author: dabdelkader
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional

class MomentumBase(ABC):
    """
    Abstract base class for momentum-based smoothing of parameter updates.
    """
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.param_keys = []
        self.smoothed_values = np.array([])
        self.initial_values = np.array([])
        self.optimal_values = np.array([])
        self.hyperparams = {}
    
    @abstractmethod
    def _load_cache(self) -> None:
        """Load internal state from disk."""
        pass
    
    @abstractmethod
    def _save_cache(self) -> None:
        """Save current state to disk."""
        pass
    
    def set_initial_values(self, values: Dict[str, float]) -> None:
        """
        Set initial parameter values (x₀).
        
        Args:
            values: Dictionary of parameter names and their initial values.
        """
        values = values.copy()
        if not self.param_keys:
            self.param_keys = list(values.keys())
        elif self.param_keys != list(values.keys()):
            raise ValueError("Parameter keys must match across iterations.")
            
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
            if values.keys() != self.initial_values_dict().keys():
                raise ValueError("Parameter keys must match between initial and optimized values.")
        
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
    
    def initial_values_dict(self) -> Dict[str, float]:
        """Convert initial values array back to dictionary."""
        if not hasattr(self, 'initial_values'):
            return {}
        return {k: float(v) for k, v in zip(self.param_keys, self.initial_values)}
    
    def optimal_values_dict(self) -> Dict[str, float]:
        """Convert optimal_values values array back to dictionary."""
        if not hasattr(self, 'optimal_values'):
            return {}
        return {k: float(v) for k, v in zip(self.param_keys, self.optimal_values)}
    
    def smoothed_values_dict(self) -> Dict[str, float]:
        """Convert smoothed_values values array back to dictionary."""
        if not hasattr(self, 'optimal_values'):
            return {}
        return {k: float(v) for k, v in zip(self.param_keys, self.smoothed_values)}