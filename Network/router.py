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
from igraph import Graph

logger = logging.getLogger(__name__)

class Router:
    """
    A class that provides routing functionalities for a network.
    """
    def __init__(self, network_file=None, link_stats_file=None, fit_vdf=True,
                 package = "igraph"):
        self.network = read_network(network_file) if network_file else None
        self.network.only_cars()

        if self.network is None:
            raise ValueError("Network data must be provided.")
        
        self.graph = self.network.as_igraph()
        self.speeds = SpeedCalculator(
            network=self.network,
            graph=self.graph,
            link_stats_file=link_stats_file,
            fit_vdf=fit_vdf
        )

        # Prepare KDTree for nearest node lookup
        self.node_positions = np.array([
            (v["x"], v["y"]) for v in self.graph.vs
        ])
        # self.node_ids = np.array(self.graph.vs["node_id"])
        # self.node_idx = np.array(self.graph.vs["node_idx"])
        self.kdtree = KDTree(self.node_positions)

        # Map node ID -> igraph index
        # self.id_to_index = dict(zip(self.graph.vs["node_id"], self.graph.vs["node_idx"]))
        # self.index_to_id = dict(zip(self.graph.vs["node_idx"], self.graph.vs["node_id"]))

    def _nearest_node(self, x, y):
        _, idx = self.kdtree.query((x, y))
        return idx
    
    def get_route(self, origin, destination):
        """
            Get the travel time between two nodes in the network.
            Parameters:
                origin (tuple): (x, y) coordinate of origin.
                destination (tuple): (x, y) coordinate of destination.
            Returns:
                float: Travel time or inf if no path found.
        """
        origin_node_idx = self._nearest_node(*origin)
        destination_node_idx = self._nearest_node(*destination)
        
        try:           
            path_length = self.graph.distances(
                source=origin_node_idx,
                target=destination_node_idx,            
                weights='free_travel_time'
            )[0][0]
            if path_length == float('inf'):
                raise ValueError(f"No path found between {origin_node_idx} and {destination_node_idx}.")

            return path_length

        except Exception:
            logger.warning(f"No path found between {origin_node_idx} and {destination_node_idx}.")
            return float('inf')

    def route_trips(self, trips):
        """
        Route multiple trips and calculate travel time.
        
        Parameters:
            trips (pd.DataFrame)
        
        Returns:
            travel times (list): List of travel times for each trip.
        """
        origins = np.array([trips.start_x.tolist(), trips.start_y.tolist()]).T
        destinations = np.array([trips.end_x.tolist(), trips.end_y.tolist()]).T
        
        _, origin_node_indices = self.kdtree.query(origins)
        _, dest_node_indices = self.kdtree.query(destinations)
   
        travel_times = []
        for origin_idx, dest_idx in zip(origin_node_indices, dest_node_indices):
            try:
                path_length = self.graph.distances(
                                    source=origin_idx,
                                    target=dest_idx,
                                    weights='free_travel_time'
                                    )[0][0]
                if path_length == float('inf'):
                    raise ValueError
                travel_times.append(path_length)
            except Exception:
                travel_times.append(float('inf'))
        
        return travel_times
