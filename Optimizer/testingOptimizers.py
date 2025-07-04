#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:17:57 2025

@author: dabdelkader
"""

import numpy as np
import matplotlib.pyplot as plt
import cma
from scipy.stats.qmc import Sobol
from scipy.spatial.distance import cdist


from scipy.stats.qmc import Sobol
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# Define a 2D function (e.g., a multimodal function)
def func_2d(x, y):
    out = np.sqrt((x-0.1)**2+(y-0.9)**2)*0.95
    out+= np.sqrt((x-0.2)**2+(y-0.1)**2)
    out+= np.sqrt((x-0.8)**2+(y-0.6)**2)*1.1
    out-= np.sqrt((x-0.4)**2+(y-0.5)**2)
    
    out = out-1
    return  -np.exp(-np.log(out))

# Create a grid of points
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)
Z = func_2d(X, Y)

# Flatten the 2D function for use with CMA-ES
def objective_function(x):
    return func_2d(x[0], x[1])

# CMA-ES Optimization
x0 = [0.5, 0.5]  # initial guess
sigma = 0.3      # initial standard deviation
lb = np.array([0,0])
ub = np.array([1,1])
max_evals = 1000

opts = {"bounds": [lb, ub], "popsize": 16, "maxfevals": max_evals}

es = cma.CMAEvolutionStrategy(x0, sigma, opts)

proposed_points = []
iteration = 0

while not es.stop():
    if iteration==0:        
        sobol_engine = Sobol(d=len(x0), scramble=True)
        sobol_samples = sobol_engine.random(n=200)
        solutions = np.clip(sobol_samples, 0.0, 1.0)  
        proposed_points.extend(solutions)                               
        es.inject(solutions, [objective_function(s) for s in solutions] )
    else:
        solutions = es.ask()
        proposed_points.extend(solutions)
        es.tell(solutions, [objective_function(s) for s in solutions])
    iteration+=1
    
# Extract x, y for plotting
proposed_points = np.array(proposed_points)
px, py = proposed_points[:, 0], proposed_points[:, 1]

xbest = es.result.xbest

# Plot the heatmap with proposed points overlaid
plt.figure(figsize=(6, 5))
heatmap = plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
plt.scatter(px, py, color='red', s=5, alpha=0.5, label='CMA-ES Samples')
plt.scatter(xbest[0], xbest[1], color='c', s=10, alpha=1, label='CMA-ES Xbest')

plt.colorbar(heatmap, label='Function Value')
plt.title("CMA-ES Optimization Samples on 2D Function")
plt.xlabel("x")
plt.ylabel("y")
#plt.legend()
plt.tight_layout()
plt.show()










# dim = len(lb)
# explored_solutions = []
# explored_objectives = []

# def scaler(x): return (x - lb) / (ub - lb)
# def back_scaler(x): return x * (ub - lb) + lb

# # Stage 1: Global exploration with Sobol
# sobol_engine = Sobol(d=dim, scramble=True)
# n_global = int(max_evals * 0.5)

# sobol_samples = sobol_engine.random(n=n_global)
# sobol_samples = np.clip(sobol_samples, 0, 1)

# X_global = np.array([s for s in sobol_samples])
# y_global = np.array([objective_function(back_scaler(x)) for x in X_global])

# explored_solutions.extend(X_global.tolist())
# explored_objectives.extend(y_global.tolist())

# # Stage 2: Identify interesting zones
# n_top = 5
# min_dist = 0.1
# top_indices_sorted = np.argsort((y_global))
# best_solution = y_global[top_indices_sorted[0]]

# top_solutions = []

# for idx in top_indices_sorted:
#     candidate = sobol_samples[idx]
#     if len(top_solutions) == 0:
#         top_solutions.append(candidate)
#     else:
#         distances = cdist([candidate], top_solutions)
#         if np.all(distances >= min_dist):
#             top_solutions.append(candidate)
#     if len(top_solutions) >= n_top or y_global[idx]>1.5*best_solution:
#         break 
# top_solutions = np.array(top_solutions)

# # Fit a simple kernel-density-like expansion around top solutions
# sobol_engine_local = Sobol(d=dim, scramble=True)
# n_local = int(n_global)
# refined_samples = []
# perturb_radius = min_dist

# while len(refined_samples) < n_local:
#     s = sobol_engine_local.random(1)[0]
#     center = top_solutions[np.random.randint(0, n_top)]
#     perturbed = np.clip(center + (s - 0.5) * 2 * perturb_radius, 0, 1)
#     refined_samples.append(perturbed)

# refined_samples = np.array(refined_samples)
# X_local = np.array([s for s in refined_samples])
# y_local = np.array([objective_function(back_scaler(x)) for x in X_local])

# explored_solutions.extend(refined_samples)
# explored_objectives.extend(y_local.tolist())

# # Stage 3: Train surrogate model
# all_samples = np.vstack([X_global, X_local])
# all_scores = np.concatenate([y_global, y_local])

# model = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=5,            # limit tree depth to avoid overfitting
#     min_samples_split=5,
#     random_state=0
# )
# model.fit(all_samples, all_scores)



# # Stage 4: Surrogate-based optimization
# n_surr = int(max_evals * 100)
# sobol_engine_final = Sobol(d=dim, scramble=True)
# test_samples = sobol_engine_final.random(n_surr)
# test_samples = np.clip(test_samples, 0, 1)
# preds = model.predict(test_samples)

# best_idx = np.argmin(preds)
# best_sample = back_scaler(test_samples[best_idx])
# best_pred = preds[best_idx]




# plt.figure(figsize=(6, 5))
# heatmap = plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
# plt.scatter(all_samples[:,0], all_samples[:,1], color='red', s=10, alpha=0.6, label='ML Samples')
# plt.scatter(best_sample[0], best_sample[1], color='c', s=20, alpha=1, label='ML Xbest')

# plt.colorbar(heatmap, label='Function Value')
# plt.title("CMA-ES Optimization Samples on 2D Function")
# plt.xlabel("x")
# plt.ylabel("y")
# #plt.legend()
# plt.tight_layout()
# plt.show()













