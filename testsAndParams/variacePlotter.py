#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:00:37 2025

@author: dabdelkader
"""


# Define your custom Loss class and BaseUtility first
from Loss.Loss import Loss
from Utilities.TourUtility import TourUtility
from Utilities.BaseUtility import BaseUtility
from Utilities.Parameters import Parameters
from Utilities.Selector import Selector
from Optimizer.OptimizersFactory import get_optimizer
from Optimizer.MomentumsFactory import create_momentum
import os
import time
import datetime
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# parameters
simulation_file = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/sim_output"

selector = "MultinomialLogit"
input_parameters = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/parameters.yml"
output_parameters = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/optimized_parameters.yml"
iteration = 10
population_sample = None

actual_mode_shares = {"car":0.35, "pt": 0.2, "bike": 0.15,"walk": 0.25,"car_passenger":0.05}
bounds = {"car.alpha_u": 0.1, "walk.alpha_u": 0.1, "bike.alpha_u": 0.1,}
metric = "js"
optimizer = "kai"

# path to files
bike_file      = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_bike.csv" 
pt_file        = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_pt.csv" 
car_file       = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_car.csv" 
walk_file      = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_walk.csv" 
tours_file     = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.detailed_utilities.csv"

Parameters.from_yaml(input_parameters)
Selector.set_selector(selector)
TourUtility.read_and_init(tours_file,
                          {"car":car_file,"pt":pt_file,"bike":bike_file,"walk":walk_file},
                          population_sample = population_sample)
       
# get and plot variance:
modes = ["car", "car_passenger", "pt", "bike", "walk"]
sample_sizes = [3000, 7000, 10000, 15000, 20000, 30000, 38200]
use_sobol = False
TourUtility.use_sobol = use_sobol
Selector.use_sobol = use_sobol


# Function to compute statistics
def get_stats(values):
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'variance': np.var(values)
    }

# Dictionary to store results
results = {mode: {'mean': [], 'median': [], 'variance': []} for mode in modes}

# Loop through different population samples
for pop_sample in sample_sizes:
    print(f"Running for population sample = {pop_sample}")
    
    TourUtility.set_population_sample(pop_sample)
    myLoss = Loss(actual_mode_shares, metric=metric)

    # Store mode shares per iteration
    mode_shares_per_run = defaultdict(list)
    for i in range(8):
        est_shares = myLoss.estimated_mode_shares
        for mode in modes:
            mode_shares_per_run[mode].append(est_shares.get(mode, 0.0))

    # Compute and store stats for each mode
    for mode in modes:
        stats = get_stats(mode_shares_per_run[mode])
        results[mode]['mean'].append(stats['mean'])
        results[mode]['median'].append(stats['median'])
        results[mode]['variance'].append(stats['variance'])

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

ylim = {'car':[0.35,0.37], 'car_passenger':[0.08,0.095], 'pt':[0.16,0.173], 
        'bike':[0.056,0.069], 'walk':[0.31,0.33]}
for ax_idx, mode in enumerate(modes):
    ax = axes[ax_idx]
    mean_values = np.array(results[mode]['mean'])
    median_values = np.array(results[mode]['median'])    
    std_values = np.sqrt(np.array(results[mode]['variance']))  # Variance -> Std

    # Plot mean as main line/points
    ax.plot(sample_sizes, mean_values, label='Mean', marker='o', linestyle='--')

    # Plot median as separate points
    ax.scatter(sample_sizes, median_values, color='red', zorder=3, label='Median')

    # Plot error bars around mean using std
    ax.errorbar(sample_sizes, mean_values, yerr=std_values, fmt='none',
                ecolor='gray', capsize=5, alpha=0.7, label=r'$\pm$ 1 Std Dev')

    # Formatting
    ax.set_title(f"{mode.capitalize()} Mode Share")
    ax.set_xlabel("Population Sample Size")
    ax.set_ylabel("Mode Share")
    ax.set_ylim(ylim[mode])
    ax.grid(True)
    ax.legend()

# Remove unused subplot
if len(modes) < len(axes):
    for ax in axes[len(modes):]:
        ax.remove()

plt.tight_layout()
plt.savefig("variance.png", dpi=100, bbox_inches="tight")



















