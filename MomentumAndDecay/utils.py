# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:49:14 2025

@author: dabdelkader
"""
import numpy as np

def to_lists(d: dict):
    """
    Recursively converts numpy arrays and np.float64 in a dictionary to lists and floats.
    """
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = to_lists(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.float64):
            out[k] = float(v)
        elif isinstance(v, list):
            # Recursively handle lists of dicts/arrays
            out[k] = [to_lists(item) if isinstance(item, (dict, np.ndarray, np.float64, list)) else item for item in v]
        else:
            out[k] = v
    return out

def to_arrays(d: dict):
    """
    Recursively converts lists in a dictionary to numpy arrays.
    """
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = to_arrays(v)
        elif isinstance(v, list):
            # Recursively handle lists of dicts/lists
            if all(isinstance(item, dict) for item in v):
                out[k] = [to_arrays(item) for item in v]
            else:
                out[k] = np.array(v)
        else:
            out[k] = v
    return out