#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dabdelkader
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import logging
from Network.network import read_network
from Network.speeds import SpeedCalculator

logger = logging.getLogger(__name__)

class Router:
    """
    A class that provides routing functionalities for a network.
    """
    def __init__(self, network_file=None, link_stats_file=None, fit_vdf=True):
        self.network = read_network(network_file) if network_file else None

        if self.network is None:
            raise ValueError("Network data must be provided.")
        
        self.graph = self.network.as_networkx()
        self.speeds = SpeedCalculator(
            network=self.network,
            graph=self.graph,
            link_stats_file=link_stats_file,
            fit_vdf=fit_vdf
        )

        # Preprocess node positions for fast nearest neighbor lookup
        self.node_positions = np.array([
            (data['x'], data['y']) for _, data in self.graph.nodes(data=True)
        ])
        self.node_ids = list(self.graph.nodes)
        self.kdtree = KDTree(self.node_positions)

    def _nearest_node(self, x, y):
        _, idx = self.kdtree.query((x, y))
        return self.node_ids[idx]

    def get_route(self, origin, destination):
        """
        Get the travel time between two nodes in the network.
        
        Parameters:
            origin (tuple): (x, y) coordinate of origin.
            destination (tuple): (x, y) coordinate of destination.
        
        Returns:
            float: Travel time or inf if no path found.
        """
        try:
            origin_node = self._nearest_node(*origin)
            destination_node = self._nearest_node(*destination)
            route = nx.shortest_path(self.graph, source=origin_node, target=destination_node, weight='travel_time')
            return sum(self.graph[u][v]['travel_time'] for u, v in zip(route[:-1], route[1:]))
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {origin_node} and {destination_node}.")
            return float('inf')

    def route_trips(self, trips):
        """
        Route multiple trips and calculate travel time.
        
        Parameters:
            trips (pd.DataFrame): Must contain 'origin' and 'destination' columns.
        
        Returns:
            travel times (list): List of travel times for each trip.
        """
        travel_times = [
            self.get_route(trip.origin, trip.destination)
            for trip in trips.itertuples(index=False)
        ]
        
        return travel_times
