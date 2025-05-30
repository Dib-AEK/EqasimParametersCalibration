#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 10:23:29 2025

@author: dabdelkader
"""

import os
import time
import argparse
import datetime
from typing import Dict

# Custom module imports (ensure these modules are in PYTHONPATH or same directory)
from Loss.Loss import Loss
from Utilities.TourUtility import TourUtility
from Utilities.BaseUtility import BaseUtility
from Utilities.Selector import Selector
from Utilities.Parameters import Parameters
from Optimizer.OptimizersFactory import get_optimizer
from Optimizer.MomentumsFactory import create_momentum
from Optimizer.BetaRateRise import BetaRateRise
from Optimizer.PopulationFactor import PopulationFactor

# Initiate the logger
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mode share optimization tool.")

    parser.add_argument("--selector", type=str, default="MaximumUtilitySelector",
                        choices=["MultinomialLogit","Maximum"],
                        help="Selector class to use (default: MaximumUtilitySelector)")
    
    parser.add_argument("--input-parameters", type=str, default="/home/dabdelkader/Work/Codes/Simulation_ch0p1/parameters.yml",
                        help="Path to input YAML parameters file.")
    
    parser.add_argument("--output-parameters", type=str, default="/home/dabdelkader/Work/Codes/Simulation_ch0p1/optimized_parameters.yml",
                        help="Path to output optimized YAML parameters file.")
    
    parser.add_argument("--variables-path", type=str, default="/home/dabdelkader/Work/Codes/Simulation_ch0p1/sim_output",
                        help="Base path to simulation outputs.")
    
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration number to read simulation files from.")
    
    parser.add_argument("--actual-mode-shares", type=str, default='car:0.35,pt:0.2,bike:0.15,walk:0.25,car_passenger:0.05',
                        help="Actual mode shares as comma-separated key:value pairs (e.g., car:0.35,pt:0.2)")
    
    parser.add_argument("--bounds", type=str, default='pt.alpha_u:2,car.alpha_u:2,walk.alpha_u:2,bike.alpha_u:2',
                        help="Parameter bounds as comma-separated key:value pairs")
    
    parser.add_argument("--metric", type=str, default="js",
                        choices=["mse", "mae", "cosine", "kl", "js", "hellinger", "tv"],
                        help="Metric to use for loss calculation (default: js)")
    
    parser.add_argument("--optimizer", type=str, default="ga",
                        choices=["ga","pso", "random", "bayesian","tpe","cmaes","spsa","adam",
                                 'Nelder-Mead','Powell', 'CG',  'BFGS', 'Newton-CG','L-BFGS-B',
                                 'TNC', 'COBYLA','COBYQA','SLSQP','trust-constr','dogleg','trust-ncg',
                                 'trust-exact','trust-krylov', "kai"],
                        help="Optimization algorithm to use")
    
    parser.add_argument("--momentum", type=str, default="ema",
                        choices=["ema","adam"],
                        help="Momentum")
    
    parser.add_argument("--beta-momentum", type=float, default=0.8, help="Momentum of the EMA")
    
    parser.add_argument("--max-evals", type=int, default=200,                        
                        help="Maximum number of evaluation of the loss function")
    
    parser.add_argument("--population-sample", type=int, default=1000000,                        
                        help="Maximum number of evaluation of the loss function")
    return parser.parse_args()


def parse_dict(input_str: str) -> Dict[str, float]:
    """Convert string like 'key1:val1,key2:val2' into a dictionary."""
    return {k: float(v) for k, v in (pair.split(":") for pair in input_str.split(","))}


def main():
    args = parse_args()

    # Parse actual mode shares and bounds
    try:
        actual_mode_shares = parse_dict(args.actual_mode_shares)
        bounds = parse_dict(args.bounds)
    except ValueError as e:
        raise ValueError(f"Error parsing input dict: {e}")

    # Paths
    sim_path = args.variables_path
    sim_iter = os.path.basename(sim_path).split('.')[-1]
    bike_file = f"{sim_path}/{sim_iter}.choice_variables_bike.csv"
    pt_file = f"{sim_path}/{sim_iter}.choice_variables_pt.csv"
    car_file = f"{sim_path}/{sim_iter}.choice_variables_car.csv"
    walk_file = f"{sim_path}/{sim_iter}.choice_variables_walk.csv"
    tours_file = f"{sim_path}/{sim_iter}.detailed_utilities.csv"

    # Validate files exist
    for path in [bike_file, pt_file, car_file, walk_file, tours_file]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file does not exist: {path}")
            
    # Get initial beta and population
    BetaRateRise.set_beta(args.beta_momentum)
    beta = BetaRateRise.get_beta(args.iteration) 

    PopulationFactor.set_population(args.population_sample)
    population = PopulationFactor.get_population(args.iteration)
            
    # Load parameters
    Parameters.from_yaml(args.input_parameters)
    Selector.set_selector(args.selector)
    
    # Initialize utility data
    
    TourUtility.read_and_init(tours_file, {"car": car_file,
                                           "pt": pt_file,
                                           "bike": bike_file,
                                           "walk": walk_file }, 
                              population_sample = population)

    # Setup loss function 
    myLoss = Loss(actual_mode_shares, metric=args.metric)    

    # Set momuntum   
    momuntum = create_momentum(momentum_type = args.momentum, 
                               momentum = beta)    
    initial_parameters = Parameters.get_parameters(bounds.keys()).copy()
    momuntum.set_initial_values(initial_parameters)
    
    # Create optimizer
    optimizer = get_optimizer(
        method=args.optimizer,
        objective_function=myLoss,
        bounds=bounds,
        max_evals=args.max_evals
    )

    # Run optimization
    logger.info(f"[{datetime.datetime.now()}] Starting optimization...")
    t0 = time.time()
    result = optimizer.optimize()
    t1 = time.time()
    dt = t1 - t0
    logger.info(f"Optimization completed in {int(dt//60)}:{int(dt%60):02d}")
    
    # Update optimal values (applying momentum)    
    optimal_parameters = Parameters.get_parameters(bounds.keys()).copy()
    momuntum.set_optimal_values(optimal_parameters)
    smoothed_optimal_values = momuntum.get_updated_values()
    Parameters.set_parameters(smoothed_optimal_values)
    
    # Save optimized parameters
    Parameters.to_yaml(args.output_parameters)
    logger.info(f"Optimized parameters saved to: {args.output_parameters}")


if __name__ == "__main__":
    main()