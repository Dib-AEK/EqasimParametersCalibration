# -*- coding: utf-8 -*-
"""
@author: dabdelkader
"""

import os
os.chdir("..")

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import jax.tree_util as tree_util

def compute_utilities(alpha, beta_time, beta_dist, travel_time, distance):
    return alpha + beta_time * travel_time 

def compute_probabilities(util_car, util_pt):
    utilities = jnp.stack([util_car, util_pt], axis=1)
    return jax.nn.softmax(utilities, axis=1)

def bin_distances(distances, bin_edges):
    bin_ids = jnp.digitize(distances, bin_edges[:-1])  # [1, num_bins]
    return bin_ids - 1  # shift to start at 0

def compute_mode_shares_soft(probs):
    return jnp.mean(probs, axis=0)

def compute_binned_mode_shares(probs, bins, num_bins):
    def bin_share(bin_id):
        in_bin = (bins == bin_id)
        bin_probs = jnp.where(in_bin[:, None], probs, 0.0)
        bin_total = in_bin.sum()
        return jnp.where(bin_total > 0,
                         bin_probs.sum(axis=0) / bin_total,
                         jnp.array([0.0, 0.0]))
    return jnp.stack([bin_share(i) for i in range(num_bins)])

def loss_fn(params, travel_time_car, distance_car, travel_time_pt, distance_pt,
            target_shares, target_binned_shares, fixed_bin_edges, num_bins=3):

    alpha_car, beta_time_car, beta_dist_car, beta_time_pt, beta_dist_pt = params
    alpha_pt = 0.0  # reference alternative

    util_car = compute_utilities(alpha_car, beta_time_car, beta_dist_car, travel_time_car, distance_car)
    util_pt = compute_utilities(alpha_pt, beta_time_pt, beta_dist_pt, travel_time_pt, distance_pt)

    probs = compute_probabilities(util_car, util_pt)

    # Global share loss
    global_shares = compute_mode_shares_soft(probs)
    loss_global = jnp.sum((global_shares - target_shares) ** 2)

    # Binned share loss
    bins = bin_distances(distance_car, fixed_bin_edges)
    binned_shares = compute_binned_mode_shares(probs, bins, num_bins)
    loss_binned = jnp.sum((binned_shares - target_binned_shares) ** 2)

    # Total loss
    loss_total = loss_global + loss_binned
    return loss_total


if __name__ == "__main__":
    N = 10000
    num_bins = 5
    target_shares = jnp.array([0.5, 0.4])
    target_binned_shares = jnp.array([
        [0.7, 0.3],
        [0.65, 0.35],
        [0.6, 0.4],
        [0.55, 0.45],
        [0.5, 0.5],
    ])

    learning_rate = 0.015
    num_steps = 3000

    key = random.PRNGKey(0)
    key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

    # Generate synthetic data
    travel_time_car_min = random.uniform(subkey1, (N,), minval=10, maxval=60)
    distance_car_km = (travel_time_car_min / 60) * 50
    distance_car_km *= random.uniform(subkey2, (N,), minval=0.6, maxval=1.5)

    travel_time_pt_min = travel_time_car_min * random.uniform(subkey3, (N,), minval=0.5, maxval=1.1)
    distance_pt_km = (travel_time_pt_min / 60) * 30
    distance_pt_km *= random.uniform(subkey4, (N,), minval=0.8, maxval=1.4)

    # Normalize input features
    travel_time_car = travel_time_car_min / 60.0  # convert to hours
    travel_time_pt = travel_time_pt_min / 60.0
    distance_car = distance_car_km
    distance_pt = distance_pt_km

    # Precompute fixed bin edges
    fixed_bin_edges = jnp.quantile(distance_car, jnp.linspace(0, 1, num_bins + 1))

    # Initialize parameters
    params = jnp.array([0.5, -0.1, -0.1, -0.1, -0.1])

    # JIT compiled gradient function
    grad_fn = jax.jit(jax.grad(loss_fn), static_argnames=["num_bins"])

    # Optimization loop
    for step in range(num_steps):
        grads = grad_fn(params, travel_time_car, distance_car,
                        travel_time_pt, distance_pt,
                        target_shares, target_binned_shares, fixed_bin_edges, num_bins)

        # Clip gradients
        grads = tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

        params -= learning_rate * grads

        if step % 50 == 0 or step == num_steps - 1 or step == 0:
            loss_val = loss_fn(params, travel_time_car, distance_car,
                               travel_time_pt, distance_pt,
                               target_shares, target_binned_shares, fixed_bin_edges, num_bins)
            print(f"Step {step}, Loss: {loss_val:.5f}, Params: {params}, Grads Sum: {jnp.sum(grads):.5f}")

    ############################## FINAL SHARES ##############################

    alpha_car, beta_time_car, beta_dist_car, beta_time_pt, beta_dist_pt = params
    alpha_pt = 0.0

    util_car = compute_utilities(alpha_car, beta_time_car, beta_dist_car, travel_time_car, distance_car)
    util_pt = compute_utilities(alpha_pt, beta_time_pt, beta_dist_pt, travel_time_pt, distance_pt)
    probs = compute_probabilities(util_car, util_pt)

    shares = compute_mode_shares_soft(probs)
    print(f"\nFinal Mode Shares:")
    print(f"  - Car: {shares[0]:.2f}")
    print(f"  - Public Transport: {shares[1]:.2f}")

    bins = bin_distances(distance_car, fixed_bin_edges)
    binned_shares = compute_binned_mode_shares(probs, bins, num_bins)

    plt.figure(figsize=(8, 4))
    plt.plot(target_binned_shares[:, 0], label="Target Car", color="blue")
    plt.plot(target_binned_shares[:, 1], label="Target PT", color="orange")
    plt.plot(binned_shares[:, 0], "--", label="Predicted Car", color="blue")
    plt.plot(binned_shares[:, 1], "--", label="Predicted PT", color="orange")
    plt.xlabel("Distance Bin")
    plt.ylabel("Mode Share")
    plt.title("Target vs Predicted Mode Shares by Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()