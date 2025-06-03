#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 14:31:51 2025

@author: dabdelkader
"""

from Loss.Loss import Loss
from Utilities.Parameters import Parameters
from Utilities.Selector import Selector
from Utilities.TourUtility import TourUtility
import numpy as np
import matplotlib.pyplot as plt
from skopt.sampler import Sobol
import seaborn as sns
import pandas as pd


# parameters
simulation_file = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/sim_output"

selector = "MultinomialLogit"
input_parameters = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/parameters.yml"
output_parameters = "/home/dabdelkader/Work/Codes/Simulation_ch0p1/optimized_parameters.yml"
iteration = 10
population_sample = 20000
beta = 0.9

actual_mode_shares = {"car":0.35, "pt": 0.2, "bike": 0.15,"walk": 0.25,"car_passenger":0.05}
bounds = {"car.alpha_u": 1, "walk.alpha_u": 1, "bike.alpha_u": 1,}
metric = "js"
optimizer = "pso"
momentum_type = "ema"

# path to files
bike_file      = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_bike.csv" 
pt_file        = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_pt.csv" 
car_file       = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_car.csv" 
walk_file      = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.choice_variables_walk.csv" 
tours_file     = f"{simulation_file}/ITERS/it.{iteration}/{iteration}.detailed_utilities.csv"


## Load data
Parameters.from_yaml(input_parameters)
Selector.set_selector(selector)
TourUtility.read_and_init(tours_file,
                          {"car":car_file,"pt":pt_file,"bike":bike_file,"walk":walk_file},
                          population_sample = population_sample)


########## Plotting the loss ##########
myLoss = Loss(actual_mode_shares, metric = metric)
def evaluate(params):
    Parameters.set_parameters(params)
    return myLoss.get_loss()


initial_parameters = Parameters.get_parameters(bounds.keys()).copy()

# Define bounds: ±1 around initial values
params_names = list(initial_parameters.keys())
lb = [initial_parameters[p] - bounds[p] for p in params_names]
ub = [initial_parameters[p] + bounds[p] for p in params_names]

# Number of samples
num_samples = 300

# Generate Sobol sequence samples
sampler = Sobol()
dimensions = [(lbi, ubi) for lbi, ubi in zip(lb, ub)]
samples = sampler.generate(dimensions, n_samples=num_samples)

# Evaluate each sample
results = []
for i, sample in enumerate(samples):
    param_dict = dict(zip(params_names, sample))
    loss = evaluate(param_dict)
    print(f"Sample {i+1}/{num_samples} - Loss: {loss:.4f}")
    results.append({"loss": loss, **param_dict})

# Convert to DataFrame
df = pd.DataFrame(results)


# Plotting
# 3D Scatter Plot (only if 3 parameters)
if len(params_names) == 3:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df[params_names[0]], df[params_names[1]], df[params_names[2]], c=df['loss'], cmap='viridis', s=50)
    plt.colorbar(sc, label='Loss')
    ax.set_xlabel(params_names[0])
    ax.set_ylabel(params_names[1])
    ax.set_zlabel(params_names[2])
    plt.title('Loss Landscape via Sobol Sequence')
    plt.show()

# Pairwise 2D Plots with Seaborn
sns.pairplot(data=df, hue='loss', palette='viridis', height=1.5, aspect=1.5, plot_kws={'alpha':0.6})
plt.suptitle('Pairwise Parameter Space with Loss', y=1.02)
plt.show()















