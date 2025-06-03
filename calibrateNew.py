#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:49:46 2025

@author: dabdelkader
"""

# main.py
import os
import time
import logging
import datetime

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
if __name__ == "__main__":    
    args = parse_args()    
    bounds = parse_dict(args.bounds)
    
    # Get the files
    files    = get_files(args)
    # get beta and populatio sample
    beta, population = get_beta_and_population(args)
    
    logger.info(f"[DEBUG] Population sample used for optimization: {population}")
    logger.info(f"[DEBUG] Beta momentum used for updating parameters: {beta}")
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

    # Optimize
    optimizer = get_optimizer(
        method=args.optimizer,
        objective_function=myLoss,
        bounds=bounds,
        max_evals=args.max_evals
    )
    
    logger.info(f"[DEBUG] Starting optimization...")
    t0 = time.time()
    result = optimizer.optimize()
    dt = time.time() - t0
    logger.info(f"[DEBUG] Optimization completed in {int(dt//60)}:{int(dt%60):02d}")
    
    #apply the momentum
    optimal_parameters = Parameters.get_parameters(bounds.keys()).copy()
    momuntum.set_optimal_values(optimal_parameters)
    smoothed_optimal_values = momuntum.get_updated_values()
    Parameters.set_parameters(smoothed_optimal_values)
    
    #Save parameters
    Parameters.to_yaml(args.output_parameters)
    logger.info(f"[DEBUG] Optimized parameters saved to: {args.output_parameters}")


# if __name__ == "__main__":
#     main()
















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
    



