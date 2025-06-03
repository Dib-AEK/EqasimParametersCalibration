#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 18:32:49 2025

@author: dabdelkader
"""

import numpy as np

class PopulationFactor:    
    _minimum_population = 8000
    _population = 10000
    _initial_rate = 0.1
    
    @classmethod
    def set_population(cls, population):        
        cls._population = population
    
    @classmethod
    def set_minimum_population(cls, minimum_population):        
        cls._minimum_population = minimum_population
    
    @classmethod
    def set_initial_rate(cls, initial_rate):        
        cls._initial_rate = initial_rate

    @classmethod
    def get_population(cls, iteration=None, maximum_iteration=100, method='linear', **kwargs):
        if iteration is None:
            return cls._population
        
        if method == 'linear':
            return cls._linear(iteration, maximum_iteration)
        elif method == 'exponential':
            return cls._exponential(iteration, maximum_iteration, **kwargs)
        elif method == 'logistic':
            return cls._logistic(iteration, maximum_iteration, **kwargs)
        elif method == 'quadratic':
            return cls._quadratic(iteration, maximum_iteration)
        elif method == 'cubic':
            return cls._cubic(iteration, maximum_iteration)
        elif method == 'sinusoidal':
            return cls._sinusoidal(iteration, maximum_iteration, **kwargs)
        elif method == 'step':
            return cls._step(iteration, maximum_iteration, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


    @classmethod
    def _linear(cls, iteration, maximum_iteration):
        alpha = min(iteration / maximum_iteration, 1)
        delta = cls._population - cls._minimum_population
        return int(cls._minimum_population + delta * alpha)

    @classmethod
    def _exponential(cls, iteration, maximum_iteration, rate=None):
        rate = rate if rate is not None else cls._initial_rate
        alpha = min(iteration / maximum_iteration, 1)
        delta = cls._population - cls._minimum_population
        pop = cls._minimum_population + delta * (1 - np.exp(-rate * iteration))
        return int(pop)

    @classmethod
    def _logistic(cls, iteration, maximum_iteration, rate=None):
        rate = rate if rate is not None else cls._initial_rate
        midpoint = maximum_iteration / 2
        delta = cls._population - cls._minimum_population
        pop = cls._minimum_population + delta / (1 + np.exp(-rate * (iteration - midpoint)))
        return int(pop)

    @classmethod
    def _quadratic(cls, iteration, maximum_iteration):
        alpha = min(iteration / maximum_iteration, 1)
        delta = cls._population - cls._minimum_population
        pop = cls._minimum_population + delta * (alpha ** 2)
        return int(pop)

    @classmethod
    def _cubic(cls, iteration, maximum_iteration):
        alpha = min(iteration / maximum_iteration, 1)
        delta = cls._population - cls._minimum_population
        pop = cls._minimum_population + delta * (alpha ** 3)
        return int(pop)

    @classmethod
    def _sinusoidal(cls, iteration, maximum_iteration, frequency=1, amplitude=None):
        alpha = min(iteration / maximum_iteration, 1)
        delta = cls._population - cls._minimum_population
        amplitude = amplitude if amplitude is not None else delta * 0.1
        mid = (cls._population + cls._minimum_population) / 2
        pop = mid + amplitude * np.sin(2 * np.pi * frequency * alpha)
        return int(pop)

    @classmethod
    def _step(cls, iteration, maximum_iteration, steps=5):
        step_size = max(1, maximum_iteration // steps)
        level = min(iteration // step_size, steps - 1)
        delta = cls._population - cls._minimum_population
        pop = cls._minimum_population + delta * ((level + 1) / steps)
        return int(pop)
