#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:47:46 2025

@author: dabdelkader
"""


import argparse
import warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mode share optimization tool.")
    parser.add_argument("--selector", type=str, default="MultinomialLogit",
                        choices=["MultinomialLogit", "Maximum"],
                        help="Selector class to use (default: MaximumUtilitySelector)")
    
    parser.add_argument("--calibrate-global-modeshare", type=bool, default=False,
                        help="Whether to calibrate the global mode share or not")
    
    parser.add_argument("--calibrate-modeshare-distribution", type=bool, default=True,
                        help="Whether to calibrate mode share distribution over distance or not")
    
    parser.add_argument("--input-parameters", type=str, default="/home/dabdelkader/Work/Codes/Simulation_ch0p1/parameters.yml",
                        help="Path to input YAML parameters file.")
    
    parser.add_argument("--output-parameters", type=str, default="/home/dabdelkader/Work/Codes/Simulation_ch0p1/optimized_parameters.yml",
                        help="Path to output optimized YAML parameters file.")
    
    parser.add_argument("--variables-path", type=str, default="/home/dabdelkader/Work/Codes/Simulation_ch0p1/sim_output/ITERS/it.1",
                        help="Base path to simulation outputs.")
    
    parser.add_argument("--eqasim-cache-path", type=str, default="/home/dabdelkader/Euler/ch-zh-synpop/cache10p100",
                        help="Base path to the cache dir of the synpop in order to get mode shares from microsensus.")
    
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration number to read simulation files from.")        
    
    parser.add_argument("--bounds", type=str,
                        default="car.alpha_u:1,walk.alpha_u:1,bike.alpha_u:1,bike.betaTravelTime_u_min:0.3,"
                                "car.betaTravelTime_u_min:0.25,walk.betaTravelTime_u_min:0.25,pt.betaLineSwitch_u:0.25,"
                                "pt.betaWaitingTime_u_min:0.25",
                        help="Parameter bounds as comma-separated key:value pairs")
    
    parser.add_argument("--metric", type=str, default="js",
                        choices=["mse", "mae", "cosine", "kl", "js", "hellinger", "tv"],
                        help="Metric to use for loss calculation")
    
    parser.add_argument("--optimizer", type=str, default="ga",
                        choices=["ga", "pso", "random", "bayesian", "tpe", "cmaes", "spsa", "adam",
                                 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
                                 'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr', 'dogleg',
                                 'trust-ncg', 'trust-exact', 'trust-krylov', "kai"],
                        help="Optimization algorithm to use")
    
    parser.add_argument("--momentum", type=str, default="ema", choices=["ema", "adam"],
                        help="Momentum")
    
    parser.add_argument("--beta-momentum", type=float, default=0.8, help="EMA momentum")
    
    parser.add_argument("--max-evals", type=int, default=200, help="Max evaluations of the loss function")
    
    parser.add_argument("--population-sample", type=int, default=10000000000,
                        help="Population sample size")
    args = parser.parse_args()
        
    return check_args(args)


def check_args(args):
    if args.calibrate_modeshare_distribution and all(["alpha" in p for p in args.bounds.split(',')]):
        warnings.warn(
               "Cannot calibrate mode shares distributions only by adjusting alpha parameters. "
               "Mode share distribution calibration is switched off."
           )
        args.calibrate_modeshare_distribution = False
        
    if args.optimizer=="kai" and args.calibrate_modeshare_distribution:
        warnings.warn(
         "Cannot use 'kai' optimizer for calibrating mode shares distributions. Switching to CMAES!"
         )
        args.optimizer = "cmaes"
        
    return args
        
        
        
        
        
        
        
        
        
        
        
        