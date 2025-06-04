#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:47:46 2025

@author: dabdelkader
"""


import argparse
import warnings
from Optimizer.PopulationFactor import PopulationFactor
from Optimizer.BetaRateRise import BetaRateRise
from utils.utils import parse_dict
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
                        default="""car.alpha_u:0.5,pt.alpha_u:0.5,bike.alpha_u:0.5,
                                   bike.betaTravelTime_u_min:0.2,car.betaTravelTime_u_min:0.2,
                                   pt.betaLineSwitch_u:0.2,pt.betaWaitingTime_u_min:0.2,
                                   pt.betaInVehicleTime_u_min:0.2,pt.betaAccessEgressTime_u_min:0.2""",
                        help="Parameter bounds as comma-separated key:value pairs")
    
    parser.add_argument("--metric", type=str, default="js",
                        choices=["mse", "mae", "cosine", "kl", "js", "hellinger", "tv"],
                        help="Metric to use for loss calculation")
    
    parser.add_argument("--optimizer", type=str, default="cmaes",
                        choices=["ga", "pso", "random", "bayesian", "tpe", "cmaes", "spsa", "adam",
                                 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
                                 'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr', 'dogleg',
                                 'trust-ncg', 'trust-exact', 'trust-krylov', "kai"],
                        help="Optimization algorithm to use")
    
    parser.add_argument("--momentum", type=str, default="ema", choices=["ema", "adam"],
                        help="Momentum")
    
    parser.add_argument("--beta-momentum", type=float, default=0.8, help="EMA momentum")
    
    parser.add_argument("--max-evals", type=int, default=150, help="Max evaluations of the loss function (per param when distribution is calibrated)")
    
    parser.add_argument("--population-sample", type=int, default=10000000000,
                        help="Population sample size")
    args = parser.parse_args()
        
    return check_args(args)


def check_args(args):
    # Turn bounds into a dict
    bounds = parse_dict(args.bounds)
    args.bounds = bounds
    
    if args.calibrate_modeshare_distribution and all(["alpha" in p for p in bounds.keys()]):
        warnings.warn(
               "Cannot calibrate mode shares distributions only by adjusting alpha parameters. "
               "Mode share distribution calibration is switched off."
           )
        args.calibrate_modeshare_distribution = False
    
    
    if args.calibrate_modeshare_distribution:
        logger.info( "Mode Share distribution is calibrated onces in 10 iterations" )
        perform_distribution_calibration = (args.iteration==2 or args.iteration%10==0)
        logger.info( f"Mode Share distribution calibration is{'' if perform_distribution_calibration else ' not'} performed in this iteration" )
        
        if perform_distribution_calibration:
            # calibrate only the distributions
            args.calibrate_global_modeshare = False
            # Set population to at least 30000
            PopulationFactor.set_minimum_population(30000)
            args.population_sample = max(args.population_sample, 30000)
            # set beta
            args.beta_momentum = 0.0 if args.iteration==2 else 0.6
            BetaRateRise.do_not_raise(True)            
            # set maximum evaluations
            args.max_evals = args.max_evals * len(bounds)            
        else:
          args.calibrate_modeshare_distribution = False
          args.calibrate_global_modeshare = True
          PopulationFactor.set_minimum_population(12000) #variance is more stable after that number
          
          logger.info( "Kai optimizer will be used in this iteration (more efficient)" )
          args.optimizer = "kai"
          # Only keep alphas
          keys = [key for key in bounds.keys() if "alpha" in key.lower()]
          bounds = {k:v for k,v in bounds.items() if k in keys}
          args.bounds = bounds      
          
          
    if args.optimizer=="kai" and args.calibrate_modeshare_distribution:
        warnings.warn(
         "Cannot use 'kai' optimizer for calibrating mode shares distributions. Switching to CMAES!"
         )
        args.optimizer = "cmaes"
    
    
    return args
        
        
        
        
        
        
        
        
        
        
        
        