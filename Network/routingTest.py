#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dabdelkader
"""
import os
os.chdir("..")

from Network.network import read_network, fast_network_reader
from Network.speeds import SpeedCalculator
from Network.router import Router

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# link_stats_file = 'testsAndParams/it.60/60.linkstats.txt.gz'
# network_file = 'testsAndParams/it.60/switzerland_network.xml.gz'
# trips_file = 'testsAndParams/it.60/60.trips.csv.gz'

network_file = "Z:\ch-zh-synpop\output0p1\queue_lastVersionEqasim\switzerland_network.xml.gz"
network = read_network(network_file)


# router = Router(network_file=network_file, link_stats_file=link_stats_file, fit_vdf=False)

# # read some trips from the file
# cols = ["person","trip_id","dep_time","trav_time","main_mode","start_x", "start_y","end_x","end_y"]
# trips = pd.read_csv(trips_file, nrows=5000, sep=";", usecols = cols)
# trips = trips[trips["main_mode"] == "car"]

# trips["trav_time"] = pd.to_timedelta(trips["trav_time"]).dt.total_seconds()

# #route the trips
# start_time = time.time()
# travel_times = router.route_trips(trips)
# elapsed_time = time.time() - start_time
# print(f"Routing {len(trips)} trips took {elapsed_time:.2f} seconds.")


# # plot the estimated travel times vs actual travel times
# plt.figure(figsize=(10, 6))
# plt.scatter(trips["trav_time"], travel_times, alpha=0.5)
# plt.plot([0, max(trips["trav_time"])], [0, max(trips["trav_time"])], color='red', linestyle='--')
# plt.xlabel("Actual Travel Time (s)")
# plt.ylabel("Estimated Travel Time (s)")
# plt.title("Estimated vs Actual Travel Times")
# plt.grid(alpha=0.3)





