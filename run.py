#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:49:46 2025

@author: dabdelkader
"""

import os
import time
import logging
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.cli import parse_args
from utils.utils import get_beta_and_population, get_files, update_number_of_runs, get_number_of_runs

from Loss.Loss import Loss
from Utilities.TourUtility import TourUtility
from Utilities.BaseUtility import BaseUtility
from Selector.Selector import Selector
from Utilities.Parameters import Parameters
from Optimizer.OptimizersFactory import get_optimizer
from MomentumAndDecay.MomentumsFactory import create_momentum
from modeShares.ChModeShares import ChModeShares
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optimizer (Python)")
starting_time = time.time()

# PParsing arguments
args = parse_args()

parameters_to_calibrate = args.bounds.keys()
logger.info(f"Calibrated parameters: {parameters_to_calibrate}")

# Get the files
files    = get_files(args)

# get beta and population
beta, population = get_beta_and_population(args)

logger.info(f"iter{args.iteration}: Population sample used for optimization: {population}")
logger.info(f"iter{args.iteration}: Beta momentum used for updating parameters: {beta}")

# Import initial parameters
Parameters.from_yaml(args.input_parameters)
logger.info(f"iter{args.iteration}: Initial parameters loaded from: {args.input_parameters}")

# Select the selector
Selector.set_selector(args.selector)

# define the Loss
mode_shares_provider = ChModeShares(args.eqasim_cache_path, cache_dir = args.optimizer_cache, overwrite=False)
myLoss = Loss(mode_shares_provider, metric=args.metric,
              objectives = args.objectives)

# Create momentum
momuntum = create_momentum(momentum_type=args.momentum, momentum=beta, cache_path=args.optimizer_cache)
initial_parameters = Parameters.get_parameters(parameters_to_calibrate).copy()
momuntum.set_initial_values(initial_parameters)

# loads the variables and utilities
TourUtility.read_and_init(**files,
                          population_sample=population,
                          eqasim_cache_dir = args.eqasim_cache_path,
                          optimizer_cache_dir = args.optimizer_cache,
                          mode_shares_provider = mode_shares_provider)

# find the optimal parameters through optimization
optimizer = get_optimizer(args,  objective_function=myLoss)
logger.info(f"iter{args.iteration}: Starting optimization...")

t0 = time.time()
result = optimizer.optimize(overwrite=False)
Parameters.set_parameters(result["params"])
dt = time.time() - t0
logger.info(f"iter{args.iteration}: Optimization completed in {int(dt//60)}:{int(dt%60):02d} minutes")

# plot the optimization process
optimizer.plot(show=False)

#apply the momentum
optimal_parameters = Parameters.get_parameters(parameters_to_calibrate).copy()
momuntum.set_optimal_values(optimal_parameters)
smoothed_optimal_values = momuntum.get_updated_values()
Parameters.set_parameters(smoothed_optimal_values)

#Save parameters
Parameters.to_yaml(args.output_parameters)
logger.info(f"iter{args.iteration}: Optimized parameters saved to: {args.output_parameters}")

#Update number of runs
update_number_of_runs(args)

# End
end_time = time.time()
logger.info(f"iter{args.iteration}: Total time taken: {datetime.timedelta(seconds=end_time - starting_time)}")




# for i in range(3):
#     Parameters.from_yaml(args.input_parameters) #just to restart parameters
#     # Selector.gumble = None
#     ########## Optimize ###########
#     optimizer = get_optimizer(args,  objective_function=myLoss)
    
#     logger.info(f"[DEBUG] Starting optimization...")
#     t0 = time.time()
#     result = optimizer.optimize()
#     dt = time.time() - t0
#     logger.info(f"[DEBUG] Optimization completed in {int(dt//60)}:{int(dt%60):02d}")
    
#     # #apply the momentum
#     # optimal_parameters = Parameters.get_parameters(bounds.keys()).copy()
#     # momuntum.set_optimal_values(optimal_parameters)
#     # smoothed_optimal_values = momuntum.get_updated_values()
#     # Parameters.set_parameters(smoothed_optimal_values)
    
#     # #Save parameters
#     # Parameters.to_yaml(args.output_parameters)
#     # logger.info(f"[DEBUG] Optimized parameters saved to: {args.output_parameters}")
#     Parameters.to_yaml(args.input_parameters.replace('.yml','_opt.yml'))
    
#     # if __name__ == "__main__":
#     #     main()
    
#     # plt.plot(myLoss.utility_time, label="utilities")
#     # plt.plot(myLoss.selector_time, label = "selector")
#     # plt.plot(myLoss.mode_share_time, label="mode_share")
#     # plt.legend()
    







# if False:
#     # plot mode shares distriution
    
#     import matplotlib.pyplot as plt
#     from cycler import cycler
#     import itertools
#     import numpy as np
    
#     by = "distance"
#     y = myLoss.get_estimated_mode_shares()
#     yt = myLoss.get_actual_mode_shares()
    
#     y, yt = y[by], yt[by]
    
#     if "distance" in by:
#         distance_bins = np.array(myLoss.distance_bins)
#         distance = (distance_bins[1:]+distance_bins[:-1])/2
#         distance[-1] = distance_bins[-2]
#     else:
#         distance=range(len(y[list(y.keys())[0]]))
        
#     # Set a base color cycle (e.g., from matplotlib's default colors)
#     default_cycler = cycler(color=plt.cm.tab10.colors)
#     plt.rcParams['axes.prop_cycle'] = default_cycler
    
#     # Get a cycling iterator
#     color_cycle = itertools.cycle(default_cycler())
    
#     fig, ax = plt.subplots(figsize=(10,6))
#     # Plotting
#     for mode in y.keys():
#         # Get next color from cycle
#         c = next(color_cycle)['color']
        
#         ax.plot(distance, y[mode],color=c,label=f'{mode} (estimated)')
#         ax.plot(distance, yt[mode], color=c, linestyle='--', label=f'{mode} (target)')
    
#     plt.grid(True)
#     plt.xlabel("Distance [km]")
#     plt.ylabel("Mode Share")
#     plt.title("Estimated vs Target Mode Shares by Distance")
    
#     # Optional: Reduce clutter in legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
#     plt.show()
    


# if False:
#     trips_path = "/home/dabdelkader/Euler/ch-zh-synpop/output10p100/queue_new/sim_calib_distribution/ITERS/it.80/80.trips.csv.gz"
#     trips = pd.read_csv(trips_path, sep=";")
    
#     trips = trips[['person', 'euclidean_distance', 'main_mode']]
#     trips = trips[trips.main_mode.isin(["car","pt","bike","walk","car_passenger"])]
    
#     trips['distance_bin'] = pd.cut(trips['euclidean_distance'],
#                                             bins=distance_bins,
#                                             labels=bin_labels, 
#                                             include_lowest=True, 
#                                             ordered=True)

#     grouped = trips.groupby(['distance_bin', 'main_mode'], observed=False).size().unstack(fill_value=0)
#     mode_shares_by_bin = grouped.div(grouped.sum(axis=1), axis=0).fillna(0)
#     y = {mode: mode_shares_by_bin[mode].tolist()
#                                          for mode in modes}
#     x = trips["main_mode"].value_counts(normalize=True)
#     # Checking coorelation between columns for public transport
    
#     import pandas as pd
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     pt = TourUtility.variables_by_mode["pt"].copy()
#     cols = ['accessEgressTime_min', 'inVehicleTime_min', 
#             'waitingTime_min','numberOfLineSwitches']
    
#     corr_matrix = pt[cols].corr(numeric_only=True)
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
#     plt.title("Correlation Matrix")
#     plt.show()


# if False:
#     # Plot parameters evolution from output dir
#     import contextlib

#     sim_dir = "/home/dabdelkader/Euler/ch-zh-synpop/output10p100/queue_new/sim_calib_distributionAndGlobal_lim/ITERS"
#     iters = sorted([d for d in os.listdir(sim_dir) if d.startswith("it.")],
#                    key=lambda x: int(x.split(".")[1]))
#     alphas = []
#     betas = []
#     for it in iters:
#         it_path = os.path.join(sim_dir, it)
#         files = os.listdir(it_path)
#         param_file = next((f for f in files if "optimized_parameters" in f), None)
    
#         if param_file:
#             file_path = os.path.join(it_path, param_file)
#             Parameters.from_yaml(file_path)
    
#             alpha_i = [Parameters.car.alpha_u,
#                        Parameters.pt.alpha_u,
#                        Parameters.walk.alpha_u,
#                        Parameters.bike.alpha_u]
#             beta_i  = [Parameters.car.betaTravelTime_u_min,
#                        Parameters.pt.betaInVehicleTime_u_min,
#                        Parameters.walk.betaTravelTime_u_min,
#                        Parameters.bike.betaTravelTime_u_min]
            
#             alphas.append(alpha_i)
#             betas.append(beta_i)
            
#     alphas = np.array(alphas)
#     betas = np.array(betas)
    
#     fig, ax = plt.subplots(1,2, figsize=(12,4))
#     modes = ['car', 'pt', 'walk', 'bike']
#     for idx, mode in enumerate(modes):
#         ax[0].plot(alphas[:, idx], label=f'alpha_{mode}')
#         ax[1].plot(betas[:, idx], label=f'beta_{mode}')
    
#     for axi, name in zip(ax, ["Alpha", "Beta"]):
#         axi.set_xlabel("Iteration")
#         axi.set_ylabel("Value")
#         axi.set_title(f"{name} Values Over Iterations")
#         axi.legend()
#         axi.grid(True)
    
#     plt.tight_layout()
#     plt.show()
        
    
    
# if False:
#     import polars as pl
#     #plot distribution of waiting time and line switch
#     # selected_tours = (TourUtility.tours.select(["eqasim_selected","trip_key"])
#     #                                    .filter(pl.col("eqasim_selected"))
#     #                                    .select(['trip_key'])
#     #                                    .explode(["trip_key"])
#     #                                    ).collect().to_pandas()
#     params = {'pt.alpha_u': 4,
#               'pt.betaLineSwitch_u':-6.0,
#               'pt.betaWaitingTime_u_min':-0.0,
#               'pt.betaInVehicleTime_u_min':-0.5,
#               'pt.betaAccessEgressTime_u_min':-0.2
#               }
#     Parameters.set_parameters(params)
    
    
#     tours = TourUtility.get_all_utilities().collect()        
#     tours = Selector.select(tours)
#     cols = ["candidate_mode", "trip_key"] 
#     selected_trips = tours.select(cols).explode(cols).filter(pl.col("candidate_mode")=="pt")
#     selected_trips = selected_trips.to_pandas()
    
#     pt_trips     = TourUtility.exploded_tours["pt"].collect().to_pandas()
#     pt_variables = TourUtility.variables_by_mode.get("pt").collect().to_pandas()

#     pt_trips = pt_trips[pt_trips.trip_key.isin(selected_trips.trip_key)]    
#     pt_trips = pt_trips.merge(pt_variables, on="trip_key", how="left")   
  
#     bin_edges = [0, 3, 6, 10, 15, 20, 1e6]
#     labels = ["0-3km","3-6km","6-10km","10-15km","15-20km","+20km"]
    
#     pt_trips = pt_trips.sort_values("euclideanDistance_km").reset_index(drop=True)
    
#     pt_trips['distance_bin'] = pd.cut(
#                         pt_trips['euclideanDistance_km'],
#                         bins=bin_edges, 
#                         labels = labels,
#                         include_lowest=True, 
#                         ordered=True)    
    
#     line_switches = pt_trips.groupby(["distance_bin"], observed=False)["numberOfLineSwitches"].mean()+1
#     access_time   = pt_trips.groupby(["distance_bin"], observed=False)["accessEgressTime_min"].mean()
#     wait_time     = pt_trips.groupby(["distance_bin"],observed=False)["waitingTime_min"].mean()
#     in_vehicle_time = pt_trips.groupby(["distance_bin"], observed=False)["inVehicleTime_min"].mean()
     
    
#     # microcensus
#     transit_path = args.eqasim_cache_path+"/data.microcensus.transit__cc73f02aef57a0f5f7c4e100ba36c3b1.p"
#     trips_path = args.eqasim_cache_path+"/data.microcensus.trips__cc73f02aef57a0f5f7c4e100ba36c3b1.p"
    
#     transit = pd.read_pickle(transit_path)
#     trips,_   = pd.read_pickle(trips_path)
#     trips = trips[["person_id", "trip_id", "crowfly_distance"]]
    
#     transit = transit.merge(trips, on = ["person_id", "trip_id"], how="left")
#     transit["crowfly_distance"] = transit["crowfly_distance"]*1e-3 #convert to km
    
#     transit['distance_bin'] = pd.cut(
#                                 transit['crowfly_distance'],
#                                 bins=bin_edges, 
#                                 labels = labels,
#                                 include_lowest=True, 
#                                 ordered=True)   
            
#     line_switches_m   = transit.groupby(["distance_bin"], observed=False)["line_switches"].mean()
#     access_time_m     = transit.groupby(["distance_bin"], observed=False)["access_egress_time"].mean()/60
#     wait_time_m       = transit.groupby(["distance_bin"], observed=False)["waiting_time"].mean()/60
#     in_vehicle_time_m = transit.groupby(["distance_bin"], observed=False)["in_vehicle_time"].mean()/60
    
    
#     # plot results

#     plt.style.use('seaborn-v0_8-whitegrid')
#     plt.rcParams.update({ 'font.size': 12,'axes.titlesize': 14,'axes.labelsize': 12,
#         'legend.fontsize': 10,'xtick.labelsize': 10,'ytick.labelsize': 10})
    
#     # Colors (simulated vs microcensus)
#     color_sim = '#1f77b4'  # muted blue
#     color_mc = '#d62728'   # red
    
#     # Plot
#     fig, axes = plt.subplots(2, 2, figsize=(12, 9))
#     fig.suptitle("Comparison: Simulated vs Microcensus (PT Trips)", fontsize=16)
    
#     # Helper function to plot each metric
#     def plot_metric(ax, mc_data, sim_data, title, ylabel, ylim=None):
#         ax.plot(mc_data.index.astype(str), mc_data.values, label='Microcensus', marker='o', color=color_mc)
#         ax.plot(sim_data.index.astype(str), sim_data.values, label='Simulated', marker='s', color=color_sim)
        
#         ax.set_title(title)
#         ax.set_ylabel(ylabel)
#         ax.set_xlabel("Trip Distance")
#         ax.grid(True, linestyle='--', alpha=0.5)
#         ax.legend()
        
#         if ylim:
#             ax.set_ylim(ylim)
    

#     plot_metric(axes[0, 0], in_vehicle_time_m, in_vehicle_time,
#                 "In-Vehicle Time","Avg Time (min)", ylim=[0, None] )
    
#     plot_metric(axes[0, 1], line_switches_m, line_switches,
#                 "Line Switches","Avg Number of Transfers",ylim=[0, None] )
    
#     plot_metric(axes[1, 0],access_time_m, access_time,
#                 "Access + Egress Time", "Avg Time (min)",ylim=[0, None])
    
#     plot_metric( axes[1, 1],wait_time_m, wait_time,
#                 "Waiting Time","Avg Time (min)", ylim=[0, None] )
    
#     for ax in axes.flat:
#         for label in ax.get_xticklabels():
#             label.set_rotation(20)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()
        
        
    
# if False:
#     l = [list(li) for li in optimizer.explored_solutions]
#     l = np.array(l)
#     o = np.array(optimizer.explored_objectives)
    
#     from sklearn.decomposition import PCA
#     import matplotlib.pyplot as plt
    
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(l)
#     l_approx = pca.inverse_transform(X_pca)
    
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=o, cmap='viridis', s=10)
#     plt.colorbar(label='f(x)')
#     plt.title('PCA of 5D Inputs Colored by f(x)')
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.show()
    
#     ### n x n figures plot
#     n_dims = l.shape[1]
#     fig, axes = plt.subplots(n_dims, n_dims, figsize=(15, 15))
#     plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
#     for i in range(n_dims):
#         for j in range(n_dims):
#             ax = axes[i, j]
#             if i == j:
#                 # Plot histogram of the variable itself
#                 ax.hist(l[:, i], bins=20, color='gray', alpha=0.7)
#             else:
#                 sc = ax.scatter(l[:, j], l[:, i], c=o, cmap='viridis', s=10)
#             if i == n_dims - 1:
#                 ax.set_xlabel(f"x{j+1}")
#             if j == 0:
#                 ax.set_ylabel(f"x{i+1}")
    
#     # Add colorbar only once
#     cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
#     cbar.set_label("f(x)")
    
#     plt.suptitle("5x5 Pairwise Input Scatter Plots Colored by Objective", fontsize=16)
#     plt.show()      
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    