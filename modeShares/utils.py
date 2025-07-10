# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:22:05 2025

@author: dabdelkader
"""
import numpy as np

def def_convert_all_to_list(x):
    
    if isinstance(x, dict):
        for k,v in x.items():
            x[k] = def_convert_all_to_list(v)
    
    if isinstance(x, np.ndarray):
        return list(x)
    
    return x
    

def clean_and_weighted_avg(group,x,weight=None, remove_outliers = True):    
    cols = [x] if weight is None else [x, weight]
    data = group[cols].dropna()

    if remove_outliers:
        q1 = data[x].quantile(0.25)
        q3 = data[x].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        data = data[(data[x] >= lower) & (data[x] <= upper)]

    if data.empty:
        return np.nan

    return np.average(data[x], weights=None if weight is None else data[weight])