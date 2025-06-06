#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:49:46 2025

@author: dabdelkader
"""

import os
import time
import logging
import datetime
import matplotlib.pyplot as plt

from utils.cli import parse_args
from utils.utils import (parse_dict, check_required_files, get_beta_and_population, 
                         get_files, get_mode_shares_distribution)

from Loss.Loss import Loss
from Utilities.TourUtility import TourUtility
from Utilities.BaseUtility import BaseUtility
from Utilities.Selector import Selector
from Utilities.Parameters import Parameters
from Optimizer.OptimizersFactory import get_optimizer
from Optimizer.MomentumsFactory import create_momentum
from modeShares.modeShares import ModeShares


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def main():
# if __name__ == "__main__":    
args = parse_args()    
bounds = args.bounds


args.max_evals = 50    
args.input_parameters = '/home/dabdelkader/Work/Codes/Simulation_ch0p1/60.optimized_parameters.yml'
args.calibrate_global_modeshare = True
args.calibrate_modeshare_distribution = True


# Get the files
files    = get_files(args)
# get beta and populatio sample
beta, population = get_beta_and_population(args)

logger.info(f"[DEBUG] iter{args.iteration}: Population sample used for optimization: {population}")
logger.info(f"[DEBUG] iter{args.iteration}: Beta momentum used for updating parameters: {beta}")
# Import initial parameters
Parameters.from_yaml(args.input_parameters)
# Select the selector
Selector.set_selector(args.selector)
# Create momentum
momuntum = create_momentum(momentum_type=args.momentum, momentum=beta)
initial_parameters = Parameters.get_parameters(bounds.keys()).copy()
momuntum.set_initial_values(initial_parameters)

# define the Loss
mode_shares_provider = ModeShares(args.eqasim_cache_path)
myLoss = Loss(mode_shares_provider, metric=args.metric,
              calibrate_global_modeshare = args.calibrate_global_modeshare,
              calibrate_modeshare_distribution =args.calibrate_modeshare_distribution )
# loads the variables and utilities
TourUtility.read_and_init(files["tours"], 
                          {"car": files["car"], "pt": files["pt"],"bike": files["bike"],
                            "walk": files["walk"], "car_passenger": files["car_passenger"]}, 
                          population_sample=population)

# myLoss.get_estimated_mode_shares()
# ########## Optimize ###########
# optimizer = get_optimizer(args,  objective_function=myLoss)

# logger.info(f"[DEBUG] Starting optimization...")
# t0 = time.time()
# result = optimizer.optimize()
# dt = time.time() - t0
# logger.info(f"[DEBUG] Optimization completed in {int(dt//60)}:{int(dt%60):02d}")

# #apply the momentum
# optimal_parameters = Parameters.get_parameters(bounds.keys()).copy()
# momuntum.set_optimal_values(optimal_parameters)
# smoothed_optimal_values = momuntum.get_updated_values()
# Parameters.set_parameters(smoothed_optimal_values)

# #Save parameters
# Parameters.to_yaml(args.output_parameters)
# logger.info(f"[DEBUG] Optimized parameters saved to: {args.output_parameters}")


# if __name__ == "__main__":
#     main()

plt.plot(myLoss.utility_time)
plt.plot(myLoss.selector_time)
plt.plot(myLoss.mode_share_time)














if False:
    import matplotlib.pyplot as plt
    from cycler import cycler
    import itertools
    import numpy as np
    
    
    x,y = myLoss.get_estimated_mode_shares()
    xt, yt = myLoss.get_actual_mode_shares()
    
    distance_bins = np.array(myLoss.distance_bins)
    distance = (distance_bins[1:]+distance_bins[:-1])/2
    distance[-1] = distance_bins[-2]
    # Set a base color cycle (e.g., from matplotlib's default colors)
    default_cycler = cycler(color=plt.cm.tab10.colors)
    plt.rcParams['axes.prop_cycle'] = default_cycler
    
    # Get a cycling iterator
    color_cycle = itertools.cycle(default_cycler())
    
    fig, ax = plt.subplots(figsize=(10,6))
    # Plotting
    for mode in y.keys():
        # Get next color from cycle
        c = next(color_cycle)['color']
        
        ax.plot(distance, y[mode],color=c,label=f'{mode} (estimated)')
        ax.plot(distance, yt[mode], color=c, linestyle='--', label=f'{mode} (target)')
    
    plt.grid(True)
    plt.xlabel("Distance [km]")
    plt.ylabel("Mode Share")
    plt.title("Estimated vs Target Mode Shares by Distance")
    
    # Optional: Reduce clutter in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.show()
    


if False:
    trips_path = "/home/dabdelkader/Euler/ch-zh-synpop/output10p100/queue_new/sim_calib_distribution/ITERS/it.80/80.trips.csv.gz"
    trips = pd.read_csv(trips_path, sep=";")
    
    trips = trips[['person', 'euclidean_distance', 'main_mode']]
    trips = trips[trips.main_mode.isin(["car","pt","bike","walk","car_passenger"])]
    
    trips['distance_bin'] = pd.cut(trips['euclidean_distance'],
                                            bins=distance_bins,
                                            labels=bin_labels, 
                                            include_lowest=True, 
                                            ordered=True)

    grouped = trips.groupby(['distance_bin', 'main_mode'], observed=False).size().unstack(fill_value=0)
    mode_shares_by_bin = grouped.div(grouped.sum(axis=1), axis=0).fillna(0)
    y = {mode: mode_shares_by_bin[mode].tolist()
                                         for mode in modes}
    x = trips["main_mode"].value_counts(normalize=True)
    # Checking coorelation between columns for public transport
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    pt = TourUtility.variables_by_mode["pt"].copy()
    cols = ['accessEgressTime_min', 'inVehicleTime_min', 
            'waitingTime_min','numberOfLineSwitches']
    
    corr_matrix = pt[cols].corr(numeric_only=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Matrix")
    plt.show()


if False:
    import contextlib

    sim_dir = "/home/dabdelkader/Euler/ch-zh-synpop/output10p100/queue_new/sim_calib_distributionAndGlobal_lim/ITERS"
    iters = sorted([d for d in os.listdir(sim_dir) if d.startswith("it.")],
                   key=lambda x: int(x.split(".")[1]))
    alphas = []
    betas = []
    for it in iters:
        it_path = os.path.join(sim_dir, it)
        files = os.listdir(it_path)
        param_file = next((f for f in files if "optimized_parameters" in f), None)
    
        if param_file:
            file_path = os.path.join(it_path, param_file)
            Parameters.from_yaml(file_path)
    
            alpha_i = [Parameters.car.alpha_u,
                       Parameters.pt.alpha_u,
                       Parameters.walk.alpha_u,
                       Parameters.bike.alpha_u]
            beta_i  = [Parameters.car.betaTravelTime_u_min,
                       Parameters.pt.betaInVehicleTime_u_min,
                       Parameters.walk.betaTravelTime_u_min,
                       Parameters.bike.betaTravelTime_u_min]
            
            alphas.append(alpha_i)
            betas.append(beta_i)
            
    alphas = np.array(alphas)
    betas = np.array(betas)
    
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    modes = ['car', 'pt', 'walk', 'bike']
    for idx, mode in enumerate(modes):
        ax[0].plot(alphas[:, idx], label=f'alpha_{mode}')
        ax[1].plot(betas[:, idx], label=f'beta_{mode}')
    
    for axi, name in zip(ax, ["Alpha", "Beta"]):
        axi.set_xlabel("Iteration")
        axi.set_ylabel("Value")
        axi.set_title(f"{name} Values Over Iterations")
        axi.legend()
        axi.grid(True)
    
    plt.tight_layout()
    plt.show()
        
    
    
    
    
    
    
    
    
    