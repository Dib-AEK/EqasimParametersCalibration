# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:47:24 2025

@author: dabdelkader
"""

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# === Define the parameter space ===
space = [Real(-5.0, 5.0, name=f'x{i}') for i in range(2)]

# === Define the black-box function ===
@use_named_args(space)
def black_box_function(**params):
    x = [params[f'x{i}'] for i in range(2)]
    return sum((xi - 1.5)**2 for xi in x)  # Example: shifted sphere

# === Run Bayesian Optimization ===
res = gp_minimize(
    func=black_box_function,
    dimensions=space,
    n_calls=100,
    n_initial_points=30,
    acq_func="EI",  # Expected Improvement
    random_state=42
)

# === Extract evaluated points ===
X = np.array(res.x_iters)
y = np.array(res.func_vals)
x0_vals = X[:, 0]
x1_vals = X[:, 1]

# === Create heatmap of x0 vs x1 ===
grid_size = 100
x0_range = np.linspace(-5, 5, grid_size)
x1_range = np.linspace(-5, 5, grid_size)
X0, X1 = np.meshgrid(x0_range, x1_range)
Z = np.zeros_like(X0)

# Evaluate function at each grid point, keeping other dims fixed
for i in range(grid_size):
    for j in range(grid_size):
        x = [X0[i, j], X1[i, j]] + [1.5] * 10  # Fix other dims
        Z[i, j] = sum((xi - 1.5)**2 for xi in x)

# === Plot the heatmap and evaluated points ===
plt.figure(figsize=(8, 6))
plt.contourf(X0, X1, Z, levels=50, cmap='viridis')
plt.colorbar(label='Objective Value')
plt.scatter(x0_vals, x1_vals, c=y, cmap='Reds', edgecolor='black', label='Sampled Points')
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Bayesian Optimization (GP) - Heatmap with Explored Points')
plt.legend()
plt.tight_layout()
plt.show()


print("\nBest value found:", res.fun)
print("Best parameters:")
for key, val in enumerate(res.x):
    print(f"  {key} = {val:.4f}")