#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:21:23 2025

@author: dabdelkader
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


simulation_file = "../simulation_output"
files = [f"ITERS/it.{i}/{i}.detailed_utilities.csv" for i in range(len(os.listdir(simulation_file+"/ITERS")))]


df = dict()
for i,file in enumerate(files):
    try:
        file_path = os.path.join(simulation_file, file)
        dfi = pd.read_csv(file_path, sep=";")
        df[i] = dfi
    except:
        pass

for i, dfi in df.items():    
    selected_dfi = dfi[dfi.selected].reset_index(drop=True)
    avg_utility_overall = dfi.utility.mean()
    avg_utility = selected_dfi.utility.mean()
    
    modes = selected_dfi.candidate_mode.str.split(',').explode().value_counts()
    mode_shares = modes / modes.sum() * 100
    
    selected_dfi['utilities'] = selected_dfi.utilities.str.split(',')
    selected_dfi['modes'] = selected_dfi.candidate_mode.str.split(',')
    selected_dfi = selected_dfi.explode(['utilities', 'modes'])
    selected_dfi['utilities'] = selected_dfi['utilities'].astype(float)
    
    car_utility = selected_dfi[selected_dfi.modes == "car"].utilities.mean()
    pt_utility = selected_dfi[selected_dfi.modes == "pt"].utilities.mean()
    
    print(
        f"Iter {i:<2} | "
        f"U = {avg_utility_overall:>5.2f} | "
        f"Us = {avg_utility:>5.2f} | "
        f"Ucar = {car_utility:>5.2f} | "
        f"Upt = {pt_utility:>5.2f} | "
        f"car = {mode_shares.get('car', 0):>5.1f}% | "
        f"pt = {mode_shares.get('pt', 0):>5.1f}%"
    )



























