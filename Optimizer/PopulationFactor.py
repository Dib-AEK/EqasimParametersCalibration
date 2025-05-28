#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 18:32:49 2025

@author: dabdelkader
"""

import numpy as np

class PopulationFactor:
    _initial_rate = 0.1
    _minimum_population = 8000
    _population = 10000
    
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
    def get_population(cls, iteration=None, maximum_iteration = 100):
       
        if iteration is None:
            return cls._population
        
        alpha = min(iteration/maximum_iteration,1)
        pop   = cls._minimum_population + (cls._population-cls._minimum_population) * alpha
        return int(pop)

