#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:01:53 2025
@author: dabdelkader
"""

import os
import json
import numpy as np
from typing import Dict, Optional, Any, Type
from abc import ABC, abstractmethod
from MomentumAndDecay.Momentum import MomentumBase


def create_momentum(momentum_type: str, *args, **kwargs) -> MomentumBase:
    """
    Factory function to create momentum instances.
    
    Args:
        momentum_type: Type of momentum ('ema' or 'adam')
        *args: Positional arguments for constructor
        *kwargs: Keyword arguments for constructor
        
    Returns:
        Instance of the specified momentum class
    """
    momentum_map = {
        'ema': ExponentialMovingAverage,
        'adam': AdamMomentum
    }
    
    momentum_type = momentum_type.lower()
    if momentum_type not in momentum_map:
        raise ValueError(f"Unsupported momentum type: {momentum_type}. "
                         f"Supported types: {list(momentum_map.keys())}")
                         
    return momentum_map[momentum_type](*args, **kwargs)


class ExponentialMovingAverage(MomentumBase):
    """
    Applies exponential moving average (EMA) to smooth parameter updates.
    This is useful when gradients are not available, and only raw parameter changes are known.
    """
    def __init__(
        self,
        cache_path: str = "./calibrationCache/ema_cache.json",
        momentum: Optional[float] = None
    ):
        super().__init__(cache_path)
        
        # Default hyperparameters
        self.default_hyperparams = {
            "momentum": 0.9
        }
        
        # Initialize hyperparameters
        self.hyperparams = {
            "momentum": momentum if momentum is not None else 
                        self.default_hyperparams["momentum"]
        }
        self.param_keys = []
        
        self.smoothed_values = np.array([])
        self.initial_values = np.array([])
        self.optimal_values = np.array([])
        # Load previous state if exists
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load smoother state from disk."""        
        
        if os.path.exists(self.cache_path) and os.path.getsize(self.cache_path) > 0:
            try:
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
                    self.param_keys = data.get("param_keys", [])
                    self.hyperparams = data.get("hyperparams", self.hyperparams)
            except Exception as e:
                print(f"[Warning] Failed to load cache: {e}")
                
    
    def _save_cache(self) -> None:
        """Save current state to disk."""
        data = {
            "param_keys": self.param_keys,            
            "hyperparams": self.hyperparams
        }
        
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _reset_state(self) -> None:
        """Reset internal state."""
        self.param_keys = []
        self.hyperparams = self.default_hyperparams.copy()
        self.smoothed_values = np.array([])
        self.initial_values = np.array([])
        self.optimal_values = np.array([])
    
    def _update_smoother(self) -> None:
        """
        Update the EMA with new parameter values.
        
        Args:
            optimized_values: New parameter values to incorporate into the average.
        """
 
        # Initialize if needed
        if len(self.initial_values) and len(self.initial_values)==len(self.optimal_values):
            beta = self.hyperparams["momentum"]
            self.smoothed_values = beta * self.initial_values + (1 - beta) * self.optimal_values
            
            self._save_cache() #don't need it for now, maybe later
        else:
            raise ValueError(f"Either initial values or optimal values are not set!")
        


class AdamMomentum(MomentumBase):
    """
    Applies Adam-style momentum to stabilize parameter updates between iterations,
    when only initial and optimized values are available (no explicit gradients).
    """
    def __init__(
        self,
        cache_path: str = "./calibrationCache/adam_cache.json",
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        epsilon: Optional[float] = None,
        learning_rate: Optional[float] = None
    ):
        super().__init__(cache_path)
        
        # Default hyperparameters
        self.default_hyperparams = {
            "beta1": 0.8,
            "beta2": 0.9,
            "epsilon": 1e-8,
            "learning_rate": 0.5
        }
        
        # Initialize hyperparameters
        self.hyperparams = {
            "beta1": beta1 if beta1 is not None else self.default_hyperparams["beta1"],
            "beta2": beta2 if beta2 is not None else self.default_hyperparams["beta2"],
            "epsilon": epsilon if epsilon is not None else self.default_hyperparams["epsilon"],
            "learning_rate": learning_rate if learning_rate is not None else 
                              self.default_hyperparams["learning_rate"]
        }
        
        # Internal state
        self.t = 0  # Time step counter
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        
        # Load previous state if exists
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load optimizer state from disk."""        
        
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
                    self.t = data.get("t", 0)
                    self.m = np.array(data["m"], dtype=np.float64) if "m" in data else None
                    self.v = np.array(data["v"], dtype=np.float64) if "v" in data else None
                    self.param_keys = data.get("param_keys", [])
                    self.hyperparams = data.get("hyperparams", self.hyperparams)
            except Exception as e:
                print(f"[Warning] Failed to load cache: {e}")
                
    
    def _save_cache(self) -> None:
        """Save current state to disk."""
        data = {
            "t": self.t,
            "m": self.m.tolist() if self.m is not None else [],
            "v": self.v.tolist() if self.v is not None else [],
            "param_keys": self.param_keys,            
            "hyperparams": self.hyperparams
        }
        
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _reset_state(self) -> None:
        """Reset internal state."""
        self.t = 0
        self.m = None
        self.v = None
        self.param_keys = []        
        self.hyperparams = self.default_hyperparams.copy()
    
    def _update_smoother(self) -> None:
        """
        Update the Adam state with new parameter values.
        
        Args:
            optimized_values: New parameter values to incorporate into the average.
        """
        # Convert to arrays
        init_vals = self.initial_values
        opt_vals = self.optimal_values
               
        # Compute pseudo-gradient: change in parameters
        pseudo_gradient = opt_vals - init_vals
        
        # Initialize moment vectors if not done yet
        if self.m is None:
            self.m = np.zeros_like(pseudo_gradient)
        if self.v is None:
            self.v = np.zeros_like(pseudo_gradient)
        
        # Update moments
        self.t += 1
        beta1 = self.hyperparams["beta1"]
        beta2 = self.hyperparams["beta2"]
        self.m = beta1 * self.m + (1 - beta1) * pseudo_gradient
        self.v = beta2 * self.v + (1 - beta2) * (pseudo_gradient ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        
        # Adam update rule
        lr = self.hyperparams["learning_rate"]
        epsilon = self.hyperparams["epsilon"]
        update_step = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Apply update to previous optimized values (smoothed)
        self.smoothed_values = init_vals + update_step
        
        # Save state
        self._save_cache()