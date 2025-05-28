#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:20:16 2025

@author: dabdelkader
"""
import numpy as np

import numpy as np


class BetaRateRise:
    _initial_beta = 0.8  # Default starting beta 
    
    @classmethod
    def set_beta(cls, beta):
        """Sets the initial beta value."""
        cls._initial_beta = beta

    @classmethod
    def get_beta(cls, iteration=None, method='step', **kwargs):
        """
        Returns beta value at given iteration using specified rise method.
        
        Parameters:
            iteration (int): Current iteration number.
            method (str): One of ['constant', 'linear', 'sigmoid', 'cosine', 'logarithmic']
            **kwargs: Additional parameters for specific methods
        
        Returns:
            float: Rising beta value for EMA at this iteration.
        """
        if iteration is None or method == 'constant':
            return cls._initial_beta
        
        if method == 'step':
            return cls._step_rise(iteration, **kwargs)
        
        if method == 'linear':
            return cls._linear_rise(iteration, **kwargs)
        
        elif method == 'sigmoid':
            return cls._sigmoid_rise(iteration, **kwargs)
        
        elif method == 'cosine':
            return cls._cosine_rise(iteration, **kwargs)
        
        elif method == 'logarithmic':
            return cls._logarithmic_rise(iteration, **kwargs)
        
        else:
            raise ValueError(f"Unknown beta rise method: {method}")

    # --- Rise Strategies ---
    @classmethod
    def _step_rise(cls, iteration, drop_interval=50):
        """Beta decreases step-wise every N iterations."""
        d = np.floor(iteration / drop_interval) + 1
        return 1 - (1 - cls._initial_beta) / d

    @classmethod
    def _linear_rise(cls, iteration, max_iter=200):
        """Linearly interpolate from initial_beta to 1.0 over max_iter steps."""
        return cls._initial_beta + (0.99 - cls._initial_beta) * (iteration / max_iter)

    @classmethod
    def _sigmoid_rise(cls, iteration, slope=0.07, midpoint=100):
        """S-shaped rise using logistic function."""
        t = iteration - midpoint
        weight = 1 / (1 + np.exp(-slope * t))
        return cls._initial_beta + (1 - cls._initial_beta) * weight

    @classmethod
    def _cosine_rise(cls, iteration, max_iter=200):
        """Smooth cosine interpolation from initial_beta to 1.0."""
        alpha = 0.5 * (1 + np.cos(np.pi * iteration / max_iter))
        return cls._initial_beta + (1 - cls._initial_beta) * (1 - alpha)

    @classmethod
    def _logarithmic_rise(cls, iteration, scale=400):
        """Fast rise early, slows down over time."""
        alpha = np.log(iteration + 1) / np.log(scale + iteration + 1)
        return cls._initial_beta + (1 - cls._initial_beta) * alpha