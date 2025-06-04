#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:53:52 2025

@author: dabdelkader
"""

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
    # momuntum.set_optimal_values(optimal_parameters)
    # smoothed_optimal_values = momuntum.get_updated_values()
    # Parameters.set_parameters(smoothed_optimal_values)
    
    #Save parameters
    Parameters.to_yaml(args.output_parameters)
    logger.info(f"[DEBUG] Optimized parameters saved to: {args.output_parameters}")


# if __name__ == "__main__":
#     main()

