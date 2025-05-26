#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:51:06 2025

@author: dabdelkader
"""

# Define your custom Loss class and BaseUtility first
from Loss.Loss import Loss
from Utilities.TourUtility import TourUtility
from Utilities.BaseUtility import BaseUtility
from Utilities.Parameters import Parameters
from Utilities.Selector import Selector
from Optimizer.Factory import get_optimizer
import os
import time
import datetime

# parameters
simulation_file = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/sim_output"

selector = "MaximumUtilitySelector"
input_parameters = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/parameters.yml"
output_parameters = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/optimized_parameters.yml"
iteration = 1

actual_mode_shares = {"car":0.35, "pt": 0.2, "bike": 0.15,"walk": 0.25,"car_passenger":0.05}
bounds = { "pt.alpha_u": 2,  "car.alpha_u": 2, "walk.alpha_u": 2, "bike.alpha_u": 2,}
metric = "js"
optimizer = "ga"

# path to files
bike_file      = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_bike.csv" 
pt_file        = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_pt.csv" 
car_file       = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_car.csv" 
walk_file      = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_walk.csv" 
tours_file     = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.detailed_utilities.csv"


Parameters.from_yaml(input_parameters)
Selector.set_selector(selector)

TourUtility.read_and_init(tours_file,{"car":car_file,"pt":pt_file,"bike":bike_file,"walk":walk_file})

myLoss = Loss(actual_mode_shares, metric = metric)


# Use the factory to create the optimizer
optimizer = get_optimizer( method=optimizer,
                           objective_function=myLoss,
                           bounds=bounds,
                           max_evals=10
                            ) 

# Run optimization
to = time.time()
result = optimizer.optimize()
t1 = time.time()
dt = t1-to
print(f"Optimization took {int(dt//60)}:{int(dt%60):02d}")


Parameters.to_yaml(output_parameters)




# {'bike': 0.14729645742697328,
#  'car': 0.31883157240522064,
#  'car_passenger': 0.08514605344934742,
#  'pt': 0.2038533250466128,
#  'walk': 0.24487259167184586}