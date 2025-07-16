#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dabdelkader
"""

import xopen
import xml.etree.ElementTree as ET
import pandas as pd
import gzip
from tqdm import tqdm
import os
import numpy as np
import warnings
import logging
import networkx as nx 
import geopandas as gpd
import shapely.geometry as shp
from Network.network import read_network

logger = logging.getLogger(__name__)


class SpeedCalculator:
    """
    A class to calculate speeds based on a volume delay function (VDF).
    
    Attributes:
        vdf_function (callable): The volume delay function used for speed calculation.
    """
    
    def __init__(self, network=None, graph = None, link_stats_file=None, fit_vdf=True):
        self.network = network
        self.link_stats = pd.read_csv(link_stats_file, sep="\t") if link_stats_file else None
        self.link_stats = self.clean_link_stats(self.link_stats)
        
        self.graph = graph if graph is not None else self.network.as_igraph()        
        
        # Parameters of the BPR function for vdf
        self.alpha, self.beta = 0.15, 4.0  # Default values for alpha and beta in the VDF

        if self.network and self.link_stats is not None and fit_vdf:
            self.fit_vdf()        
         
    def clean_link_stats(self, df):        
        cols = ["LINK","LENGTH","FREESPEED","CAPACITY",
                *[f"HRS{i}-{i+1}avg" for i in range(24)],
                *[f"TRAVELTIME{i}-{i+1}avg" for i in range(24)]]        
        new_cols = [col.lower() for col in cols]
        cols_rename = dict(zip(cols,new_cols))
        
        df = df[cols].rename(columns=cols_rename)
        df = df.rename(columns={"link":"link_id"})
        df = df.astype({'link_id':'str',
                        'length':float,
                        'freespeed':float,
                        'capacity':float})
        # keep only links that are in the network file
        df = df[df.link_id.isin(self.network.links.link_id.unique())].reset_index(drop=True)
        return df
        
        
    
    def set_speed(self, G, df):
        """
        Set the speed attribute for each link in the network based on the provided DataFrame.
        
        Parameters:
            G (networkx.DiGraph): The directed graph representing the network.
            df (pd.DataFrame): DataFrame containing 'link_id' and 'freespeed' columns.
        
        Returns:
            None: The function modifies the graph in place.
        """
        logger.info("Setting speed attributes for links in the network")
        for idx, row in df.iterrows():
            if row['link_id'] in G:
                G.nodes[row['link_id']]['speed'] = row['speed']        

    def calculate_speed(self, df):
        """
        Calculate the speed for each link in the network based on the freespeed and length attributes, and the volume of traffic using the volume delay functions.
       
        Parameters:
            df (pd.DataFrame): DataFrame containing 'link_id', 'freespeed', 'length', 'volume', and 'capacity' columns.
        """
        logger.info("Calculating speed for links in the network")
        func = self.vdf_function
        df['speed'] = df.apply(lambda row: func(row['freespeed'], row['length'], row['volume'], row["capacity"]), axis=1)
        return df
    
    def bpr(self, freespeed, length, volume, capacity):
        """
        A function that takes freespeed, length, and volume as inputs and returns the speed.
        """
        alpha, beta = self.get_alpha_beta()         
 
        to = length / freespeed  # time to traverse the link at freespeed
        t = to * (1 + alpha * (volume / capacity) ** beta)
        return min(freespeed, length / t)  # speed = length / time        


    def get_alpha_beta(self):
        """
        Returns the alpha and beta parameters used in the volume delay function (VDF).
        
        Returns:
            tuple: A tuple containing the alpha and beta parameters.
        """
        # I need to implimnet a method that estimates alpha and beta based on the network data.
        # For now, we return some default values.
        return self.alpha, self.beta
        
    def fit_vdf(self):
        """
        Fit the volume delay function (VDF) using the network and link statistics.
        
        This method calculates the freespeed and length for each link in the network
        and prepares the DataFrame for speed calculation.
        """
        logger.info("Fitting volume delay function (VDF) using network and link statistics")
        
        if self.network is None or self.link_stats is None:
            raise ValueError("Network or link statistics data is not available.")
        
        # Prepare DataFrame with necessary columns
        df = self.link_stats.copy()
        df["free_travel_time"] = df["length"] / df["freespeed"]
        
        travel_time_cols = [f"traveltime{i}-{i+1}avg" for i in range(24)]
        volume_cols = [f"hrs{i}-{i+1}avg" for i in range(24)]
    
        df_t = pd.concat([df[['link_id',col]].rename(columns={col:'travel_time'}) for col in travel_time_cols])
        df_t["volume"] = pd.concat([df[[col]].rename(columns={col:'volume'})  for col in volume_cols])
        
        # filter out freeflow links and highly congested links
        keep = (df_t.volume>5)&(df_t.travel_time>3)&(df_t.travel_time<600)
        df_t = df_t[keep]        
        del keep
        
        # Merge everything
        df   = df[['link_id', 'freespeed', 'length', 'capacity', 'free_travel_time']]               
        df   = df_t.merge(df, on='link_id', how='left')        
        del df_t

        # filter out small links
        df = df[df.length>35]
        
        # filter out very small speeds and no congested points
        actual_speed = (df["length"]/df['travel_time'])
        df = df[(actual_speed>10/3.6)&(actual_speed<df.freespeed)].reset_index(drop=True)
        
        # Optimize alpha and beta using least squares
        from scipy.optimize import minimize
        from sklearn.metrics import mean_squared_error, r2_score
        
        n_points = len(df)        
        def loss(x):
            alpha, beta = x
            estimated_travel_time = df['free_travel_time'] * (1 + alpha * (df['volume'] / (df['capacity']*0.1)) ** beta)
            matsim_travel_time = df['travel_time']
            return mean_squared_error(matsim_travel_time, 
                                      estimated_travel_time)
        
        res = minimize(loss, 
                       x0=[self.alpha, self.beta], 
                       bounds=((0, None), (0, None)), 
                       method='L-BFGS-B')
        
        self.alpha, self.beta = res.x[0], res.x[1]
        logger.info(f"Fitted VDF parameters: alpha={self.alpha}, beta={self.beta}")

