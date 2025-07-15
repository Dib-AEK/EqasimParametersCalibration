# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:41:25 2025

@author: dabdelkader
"""

import optuna
import numpy as np
import matplotlib.pyplot as plt

# === Define the black-box function ===
def black_box_function(x):
    return sum((xi - 1.5)**2 for xi in x)  # Example: shifted sphere function

# === Define the Optuna objective ===
def objective(trial):
    x = [trial.suggest_float(f'x{i}', -5.0, 5.0) for i in range(2)]
    return black_box_function(x)

# === Run the optimization ===
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)

# === Extract trial results ===
x0_vals = []
x1_vals = []
objective_vals = []

for t in study.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        x0_vals.append(t.params['x0'])
        x1_vals.append(t.params['x1'])
        objective_vals.append(t.value)

# === Plot a heatmap of x0 vs x1 ===
grid_size = 100
x0_range = np.linspace(-5, 5, grid_size)
x1_range = np.linspace(-5, 5, grid_size)
X0, X1 = np.meshgrid(x0_range, x1_range)
Z = np.zeros_like(X0)

# Evaluate the black-box function at each point in the 2D grid
for i in range(grid_size):
    for j in range(grid_size):
        x = [X0[i, j], X1[i, j]] + [1.5]*10  # Fix other dimensions to 1.5
        Z[i, j] = black_box_function(x)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(X0, X1, Z, levels=50, cmap='viridis')
plt.colorbar(label='Objective Value')

# Overlay explored points
plt.scatter(x0_vals, x1_vals, c=objective_vals, cmap='Reds', edgecolor='black', label='Sampled Points')
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Explored Solutions (x0 vs x1) + Objective Heatmap")
plt.legend()
plt.tight_layout()
plt.show()

# Best value
print("\nBest value found:", study.best_value)
print("Best parameters:")
for key, val in study.best_params.items():
    print(f"  {key} = {val:.4f}")