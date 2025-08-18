#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:47:46 2025

@author: dabdelkader
"""


import argparse
import os
import warnings
from utils.utils import parse_dict, parse_list, check_if_files_exists
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mode share optimization tool.")
    parser.add_argument("--selector", type=str, default="MultinomialLogit",
                        choices=["MultinomialLogit", "Maximum"],
                        help="Selector class to use (default: MaximumUtilitySelector)")
    
    parser.add_argument("--input-parameters", type=str, default='testsAndParams/modeChoiceParameters.yml',
                        help="Path to input YAML parameters file.")
    
    parser.add_argument("--output-parameters", type=str, default='testsAndParams/modeChoiceOptimizedParameters.yml',
                        help="Path to output optimized YAML parameters file.")
    
    parser.add_argument("--variables-path", type=str, default='testsAndParams/it.60',
                        help="Base path to simulation outputs.")
    
    parser.add_argument("--eqasim-cache-path", type=str, default="Z:/ch-zh-synpop/cache10p100",
                        help="Base path to the cache dir of the synpop in order to get mode shares from microsensus.")
    
    parser.add_argument("--iteration", type=int, default=60,
                        help="Iteration number to read simulation files from.")        
    
    parser.add_argument("--bounds", type=str,
                        default="""car.alpha_u:4.0,
                                   walk.alpha_u:3.0,
                                   bike.alpha_u:3.0,
                                   car.betaTravelTime_u_min:0.5,
                                   walk.betaTravelTime_u_min:0.5,
                                   bike.betaTravelTime_u_min:0.5,
                                   pt.betaInVehicleTime_u_min:0.5                                                                  
                                   """,
                        help="Parameter bounds as comma-separated key:value pairs")
    
    parser.add_argument("--metric", type=str, default="mse",
                        choices=["mse", "mae", "cosine", "kl", "js", "hellinger", "tv"],
                        help="Metric to use for loss calculation")
    
    parser.add_argument("--optimizer", type=str, default="cmaes",
                        choices=["ga", "pso", "random", "bayesian", "tpe", "cmaes", "spsa", "adam",
                                 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
                                 'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr', 'dogleg',
                                 'trust-ncg', 'trust-exact', 'trust-krylov', 'dual_annealing', "kai"],
                        help="Optimization algorithm to use")
    
    parser.add_argument("--momentum", type=str, default="ema", choices=["ema", "adam"],
                        help="Momentum")
    
    parser.add_argument("--beta-momentum", type=float, default=0.8, help="EMA momentum")
    
    parser.add_argument("--max-evals", type=int, default=4000, help="Max evaluations of the loss function (per param when distribution is calibrated)")
    
    parser.add_argument("--population-sample", type=int, default=10000000000,
                        help="Population sample size")
    
    parser.add_argument("--objectives", type=str, default="global,distance,mode_distance",
                        help="Objectives to optimize")
    
    parser.add_argument("--optimizer-cache", type=str, default="optimizerCache")
    return check_and_validate_args(parser.parse_args())





def check_and_validate_args(args):
    # Turn bounds into a dict and objectives into list
    args.bounds = parse_dict(args.bounds)
    args.objectives = parse_list(args.objectives)     
    # check if there is at least one objective
    if not args.objectives:
        raise ValueError("At least one objective must be specified for the optimizer.")

    # if we only want to calibrate the alphas, there is no need to use expensive cmaes
    if all(["alpha" in p for p in args.bounds.keys()]):
        warnings.warn(
            "Since only alpha parameters are being calibrated, using a simpler optimization algorithm. Moreover, "
            "it would be faster if you use the fast_calibration module in eqasim."
        )
        args.optimizer = "kai"
    else:
        # here it means there are some betas, so switch to cmaes if optimizer is kai
        if args.optimizer == "kai":
            warnings.warn("Switching optimizer from 'kai' to 'cmaes' due to presence of beta parameters to be calibrated.")
            args.optimizer = "cmaes"


    # if we want to calibrate some of the distributions too, we need to use cmaes, it is better:
    if args.optimizer == "kai" and (args.objectives != ["global"]):
        warnings.warn(
            "Cannot use 'kai' optimizer for calibrating mode shares distributions. Calibrating only the global mode shares!"
        )
        args.objectives = ["global"]
    

    # if optimizer path doesn't exist, create it
    if not os.path.exists(args.optimizer_cache):
        os.makedirs(args.optimizer_cache)

    if "," in args.variables_path:
        args.variables_path = args.variables_path.split(",")

    # verify if the different paths provided exists
    list_of_paths = [args.input_parameters, args.eqasim_cache_path]
    list_of_paths.extend(args.variables_path)
    check_if_files_exists(list_of_paths)
    return args
        
        
        
        
        
        
        
        
        
        
        
        