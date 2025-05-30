#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:51:06 2025

@author: dabdelkader
"""

from Loss.Loss import Loss
from Utilities.TourUtility import TourUtility
from Utilities.BaseUtility import BaseUtility
from Utilities.Parameters import Parameters
from Utilities.Selector import Selector
from Optimizer.OptimizersFactory import get_optimizer
from Optimizer.MomentumsFactory import create_momentum
from Optimizer.BetaRateRise import BetaRateRise
from Optimizer.PopulationFactor import PopulationFactor
import os
import time
import datetime

# parameters
simulation_file = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/sim_output"

selector = "MultinomialLogit"
input_parameters = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/parameters.yml"
output_parameters = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/optimized_parameters.yml"
iteration = 10
population_sample = 20000
beta = 0.9
max_evals = 150

actual_mode_shares = {"car":0.35, "pt": 0.2, "bike": 0.15,"walk": 0.25,"car_passenger":0.05}
bounds = {"car.alpha_u": 1, "walk.alpha_u": 1, "bike.alpha_u": 1,}
metric = "js"
optimizer = "cmaes"
momentum_type = "ema"

# path to files
bike_file      = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_bike.csv" 
pt_file        = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_pt.csv" 
car_file       = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_car.csv" 
walk_file      = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_walk.csv" 
tours_file     = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.detailed_utilities.csv"

# Get initial beta and population
BetaRateRise.set_beta(beta)
beta = BetaRateRise.get_beta(iteration) # allows to variate beta

PopulationFactor.set_population(population_sample)
population = PopulationFactor.get_population(iteration)
population = 15000

# start loadiong data
Parameters.from_yaml(input_parameters)
Selector.set_selector(selector)
TourUtility.read_and_init(tours_file,
                          {"car":car_file,"pt":pt_file,"bike":bike_file,"walk":walk_file},
                          population_sample = population)

# Get loss and momuntum
myLoss = Loss(actual_mode_shares, metric = metric)
momuntum = create_momentum(momentum_type, momentum = beta)

# Set initial values for momuntum
initial_parameters = Parameters.get_parameters(bounds.keys()).copy()
momuntum.set_initial_values(initial_parameters)

# Use the factory to create the optimizer
optimizer = get_optimizer( method=optimizer,
                           objective_function=myLoss,
                           bounds=bounds,
                           max_evals=max_evals
                            ) 


# Run optimization
to = time.time()
result = optimizer.optimize()
t1 = time.time()
dt = t1-to
print(f"Optimization took {int(dt//60)}:{int(dt%60):02d}")

# Sooth parametrs variation (momuntum)
optimal_parameters = Parameters.get_parameters(bounds.keys()).copy()
momuntum.set_optimal_values(optimal_parameters)
smoothed_optimal_values = momuntum.get_updated_values()

# Parameters.set_parameters(smoothed_optimal_values)

# Save to yaml
Parameters.to_yaml(output_parameters)

