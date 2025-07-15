#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dabdelkader
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.covariance import LedoitWolf  # Better covariance estimation
import scipy.stats.qmc as qmc  # Faster deterministic Sobol sequence


class AdaptiveOptimizer:
    def __init__(self, func, bounds, n_init=1000, percentile=10, n_iter=100,
                 samples_per_iter=100, cluster_tol=0.05, min_std=1e-3,
                 patience=5, tol=1e-6, verbose=False):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = self.bounds.shape[0]
        self.n_init = n_init
        self.percentile = percentile
        self.n_iter = n_iter
        self.samples_per_iter = samples_per_iter
        self.cluster_tol = cluster_tol
        self.min_std = min_std
        self.patience = patience  # For early stopping
        self.tol = tol
        self.verbose = verbose

        # Store history
        self.all_samples = []
        self.all_values = []
        self.best_value = np.inf
        self.no_improvement_count = 0

    def _scale(self, X):  # [0,1]^d -> bounds
        return self.bounds[:, 0] + X * (self.bounds[:, 1] - self.bounds[:, 0])

    def _rescale(self, X):  # bounds -> [0,1]^d
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _get_interest_regions(self, X, y):
        threshold = np.percentile(y, self.percentile)
        good_indices = np.where(y <= threshold)[0]
        X_good = X[good_indices]
    
        if len(X_good) == 0:
            # Fallback: uniform sampling
            mean = np.random.rand(self.dim)
            cov = np.eye(self.dim) * self.cluster_tol ** 2
            return [mean], [cov], [1.0]
    
        if len(X_good) == 1:
            mean = X_good[0]
            cov = np.eye(self.dim) * self.cluster_tol ** 2
            return [mean], [cov], [1.0]
    
        # Compute pairwise distances and do hierarchical clustering
        dist_matrix = squareform(pdist(X_good))
        clusters = fcluster(linkage(dist_matrix), t=self.cluster_tol, criterion='distance')
    
        unique_clusters, counts = np.unique(clusters, return_counts=True)
    
        centers = []
        covariances = []
        weights = []
    
        for c in unique_clusters:
            cluster_points = X_good[clusters == c]
            if len(cluster_points) < 2:
                continue
    
            # Estimate covariance using Ledoit-Wolf shrinkage estimator for stability
            try:
                cov_estimator = LedoitWolf()
                cov_estimator.fit(cluster_points)
                cov = cov_estimator.covariance_
            except Exception:
                cov = np.cov(cluster_points.T)
    
            # Add small noise to ensure positive definiteness
            cov += np.eye(self.dim) * 1e-8
    
            mean = np.mean(cluster_points, axis=0)
            quality = 1.0 / (np.median(y[good_indices][clusters == c]) + 1e-8)
            centers.append(mean)
            covariances.append(cov)
            weights.append(quality)
    
        if not centers:
            # If no valid clusters (all too small), fall back to a few random directions
            if self.verbose:
                print("No valid clusters found; falling back to diversified sampling.")
            mean = np.mean(X_good, axis=0)
            cov = np.eye(self.dim) * self.cluster_tol ** 2
            perturbations = np.random.randn(3, self.dim) * self.cluster_tol
            centers = [mean + p for p in perturbations]
            covariances = [cov for _ in centers]
            weights = [1.0 / len(centers)] * len(centers)
    
        else:
            # Normalize weights
            weights = np.array(weights)
            weights /= weights.sum()
    
        return centers, covariances, weights

    def _build_distribution(self, centers, covs, weights):
        def sampler(n):
            samples = []
            for center, cov, w in zip(centers, covs, weights):
                count = max(1, int(n * w))
                try:
                    s = multivariate_normal.rvs(mean=center, cov=cov, size=count)
                except np.linalg.LinAlgError:
                    s = np.random.multivariate_normal(center, cov, size=count)
                s = np.clip(s, 0, 1)
                samples.append(s)
            return np.vstack(samples)
        return sampler

    def optimize(self):
        # Initial sampling
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X = sampler.random(n=self.n_init)
        X_scaled = self._scale(X)
        y = np.apply_along_axis(self.func, 1, X_scaled)

        self.all_samples.append(X_scaled)
        self.all_values.append(y)

        best_idx = np.argmin(y)
        best_x = X_scaled[best_idx]
        best_y = y[best_idx]

        if self.verbose:
            print(f"Iteration 0: Best value = {best_y:.6f}")

        for i in range(self.n_iter):
            X_all = np.vstack(self.all_samples)
            y_all = np.concatenate(self.all_values)

            # Resample from promising regions
            centers, covs, weights = self._get_interest_regions(self._rescale(X_all), y_all)
            dist_sampler = self._build_distribution(centers, covs, weights)

            X_new = dist_sampler(self.samples_per_iter)
            X_new = np.clip(X_new, 0, 1)
            X_scaled_new = self._scale(X_new)
            y_new = np.apply_along_axis(self.func, 1, X_scaled_new)

            self.all_samples.append(X_scaled_new)
            self.all_values.append(y_new)

            current_best = y_new.min()
            if current_best < best_y - self.tol:
                best_idx = np.argmin(y_new)
                best_x = X_scaled_new[best_idx]
                best_y = current_best
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.verbose:
                print(f"Iteration {i+1}: Best value = {best_y:.6f}, "
                      f"No improvement count: {self.no_improvement_count}")

            if self.no_improvement_count >= self.patience:
                if self.verbose:
                    print("Early stopping due to no improvement.")
                break

        return best_x, best_y
    


def plot_2d_function_with_samples(func, bounds, samples):
    # Create a meshgrid over the input domain
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func([x_, y_]) for x_, y_ in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Function value')

    # Plot sampled points
    if samples is not None:
        samples = np.vstack(samples)
        plt.scatter(samples[:, 0], samples[:, 1], color='red', s=3, alpha=0.5, label='Sampled Points')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("2D Function Heatmap with Sampled Points")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# Ackley function (2D)
def ackley(x):
    x = np.asarray(x)
    a = 5
    b = 0.4
    c = 0.6 * np.pi
    d = len(x)
    sum_sq = np.sum(x ** 2)
    cos_cx = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(cos_cx / d) + a + np.exp(1)

optimizer = AdaptiveOptimizer(func=ackley, bounds=[[-5, 5], [-5, 5]])
best_x, best_y = optimizer.optimize()
print("Best x:", best_x)
print("Best y:", best_y)

plot_2d_function_with_samples(ackley, bounds=[[-5, 5], [-5, 5]], samples=optimizer.all_samples)