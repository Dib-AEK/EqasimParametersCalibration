#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:49:46 2025

@author: dabdelkader
"""
from utils.cli import parse_args
# Parsing arguments (# This sould be the first thing, to switch between ch and ch_cmdp utilitis)
args = parse_args()

import os
import time
import logging
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.utils import get_beta_and_population, get_files, update_number_of_runs, get_number_of_runs
from Loss.Loss import Loss
from Utilities.TourUtility import TourUtility
from Utilities.BaseUtility import BaseUtility
from Selector.Selector import Selector
from Utilities.Parameters import Parameters
from Optimizer.OptimizersFactory import get_optimizer, get_optimal_alphas
from MomentumAndDecay.MomentumsFactory import create_momentum
from modeShares.ChModeShares import ChModeShares
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optimizer (Python)")
starting_time = time.time()



parameters_to_calibrate = args.bounds.keys()
logger.info(f"Calibrated parameters: {parameters_to_calibrate}")

# Get the files
files    = get_files(args)

# get beta and population
beta, population = get_beta_and_population(args)

logger.info(f"iter{args.iteration}: Population sample used for optimization: {population}")
logger.info(f"iter{args.iteration}: Beta momentum used for updating parameters: {beta}")

# Import initial parameters
Parameters.from_yaml(args.input_parameters)
logger.info(f"iter{args.iteration}: Initial parameters loaded from: {args.input_parameters}")

# Select the selector
Selector.set_selector(args.selector)

# define the Loss
mode_shares_provider = ChModeShares(args.eqasim_cache_path, cache_dir = args.optimizer_cache, 
                                    overwrite=False, distance_bins=args.distance_bins)
myLoss = Loss(mode_shares_provider, metric=args.metric,
              objectives = args.objectives, modes_in_loss=args.modes_in_loss)

# Create momentum
momuntum = create_momentum(momentum_type=args.momentum, momentum=beta, cache_path=args.optimizer_cache)
initial_parameters = Parameters.get_parameters(parameters_to_calibrate).copy()
momuntum.set_initial_values(initial_parameters)

# loads the variables and utilities
TourUtility.read_and_init(**files,
                          population_sample=population,
                          eqasim_cache_dir = args.eqasim_cache_path,
                          optimizer_cache_dir = args.optimizer_cache,
                          mode_shares_provider = mode_shares_provider)

# check if the utilities that are implimented are correct (only check the first 100)
tours_util = TourUtility.tours.collect()["eqasim_utility"]
all_util = TourUtility.get_all_utilities().collect()["utility"]
assert np.allclose(tours_util[:100], all_util[:100], atol=5e-3), "The utilities that are estimated do not match the actual ones"


# find the optimal parameters through optimization
optimizer = get_optimizer(args,  objective_function=myLoss)
logger.info(f"iter{args.iteration}: Starting optimization...")

t0 = time.time()
result = optimizer.optimize(overwrite=False, matsim_iteration = args.iteration)
Parameters.set_parameters(result["params"])
dt = time.time() - t0
logger.info(f"iter{args.iteration}: Optimization completed in {int(dt//60)}:{int(dt%60):02d} minutes")

# Find rapidely closest alpha values (ensure global mode shares are correct)
_ = get_optimal_alphas(args, myLoss)

# plot the optimization process
optimizer.plot(show=False)

# Apply the momentum
optimal_parameters = Parameters.get_parameters(parameters_to_calibrate).copy()
momuntum.set_optimal_values(optimal_parameters)
smoothed_optimal_values = momuntum.get_updated_values()
Parameters.set_parameters(smoothed_optimal_values)

# Save parameters
Parameters.to_yaml(args.output_parameters)
logger.info(f"iter{args.iteration}: Optimized parameters saved to: {args.output_parameters}")

# Update number of runs
update_number_of_runs(args)

# End
end_time = time.time()
logger.info(f"iter{args.iteration}: Total time taken: {datetime.timedelta(seconds=end_time - starting_time)}")


