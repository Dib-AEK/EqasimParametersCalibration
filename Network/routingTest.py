from network import read_network
from speeds import SpeedCalculator
from router import Router

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


link_stats_file = 'link_stats.csv'
network_file = 'network.xml'

router = Router(network_file=network_file, link_stats_file=link_stats_file, fit_vdf=False)

origins = [(1.0, 2.0), (3.0, 4.0)]
destinations = [(5.0, 6.0), (7.0, 8.0)]

travel_time = router.get_route(origins[0], destinations[0])
print(f"Travel time from {origins[0]} to {destinations[0]}: {travel_time}")

travel_time = router.get_route(origins[1], destinations[1])
print(f"Travel time from {origins[1]} to {destinations[1]}: {travel_time}")

