# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:54:01 2025

@author: dabdelkader
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from Utilities.BaseUtility import BaseUtility
from abc import ABC, abstractmethod


class Losses(ABC):
    
    def __init__(self, metric="js"): 
        self.metric = metric

    def get_loss(self, parameters=None):
        if parameters is not None:
            BaseUtility.set_parameters(parameters)
        return self._get_loss()
    
    @property
    def estimated_mode_shares(self):
        return self.get_estimated_mode_shares()
    
    @abstractmethod
    def get_estimated_mode_shares(self):
        """Must be implemented in subclass."""
        pass
    
    @property
    def actual_mode_shares(self):
        return self.get_actual_mode_shares()
    
    @abstractmethod
    def get_actual_mode_shares(self):
        """Must be implemented in subclass."""
        pass
    
    @abstractmethod
    def _get_loss(self):
        """Must be implemented in subclass."""
        pass

    @abstractmethod
    def _vectors(self):
        """Must be implemented in subclass."""
        pass

    def get_loss_func(self, metric):
        if metric == "mse":
            func= self.mse()
        elif metric == "mae":
            func= self.mae()
        elif metric == "mape":
            func= self.mape()        
        elif metric in {"cosine", "cosine_similarity"}:
            func=self.cosine_similarity()
        elif metric in {"kl", "kl_divergence"}:
            func= self.kl_divergence()
        elif metric in {"js", "js_divergence"}:
            func= self.js_divergence()
        elif metric in {"hellinger", "hellinger_distance"}:
            func= self.hellinger_distance()
        elif metric in {"tv", "total_variation", "total_variation_distance", "l_distance"}:
            func= self.total_variation_distance()
        elif metric in {"ll", "log_likelihood"}:
            func= self.log_likelihood()
        else:
            raise ValueError(f"Unknown loss metric: '{self.metric}'")
        
        epsilon = 1e-2 #less then 1% are not considered
        def loss_func(x,y):
            sel = (x>epsilon)
            return func(x[sel],y[sel])
        
        return loss_func


    def mse(self):        
        return mean_squared_error

    def mae(self):
        return mean_absolute_error
    
    def mape(self):
        mean_absolute_percentage_error
    
    def cosine_similarity(self):
        return lambda x, y: cosine_similarity([x], [y])[0, 0]
        
    def kl_divergence(self):
        return lambda x, y: np.sum(rel_entr(x, y + 1e-12))  
    
    def js_divergence(self):
        return jensenshannon
    
    def hellinger_distance(self):
        return lambda x, y: np.sqrt(np.sum((np.sqrt(x) - np.sqrt(y)) ** 2)) / np.sqrt(2)
    
    def total_variation_distance(self):
        return lambda x, y: np.sum(np.abs(x - y))

    def log_likelihood(self):
        def func(x, y):
            actual_probs = np.clip(x, 1e-6, 1)
            actual_probs /= actual_probs.sum()
            pred_probs = np.clip(y, 1e-6, 1)
            pred_probs /= pred_probs.sum()
            logL = np.sum(actual_probs * np.log(pred_probs))
            return -logL
        return func

    
    
    
    
    
    
    
    
    