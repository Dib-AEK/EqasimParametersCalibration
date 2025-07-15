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
import igraph as ig

logger = logging.getLogger(__name__)

"""
This script is designed to read MATSim network files. It also include a function that simplifies the network, 
by removing nodes that do not represent intersection and where links attributes do not change.

It is partially based on the implementation from:
https://github.com/matsim-vsp/matsim-python-tools/blob/master/matsim/Network.py

"""

class Network:

    _crsTag = 'coordinateReferenceSystem'

    def __init__(self, nodes, links, node_attrs, link_attrs, net_attrs=None):
        self.nodes = nodes
        self.links = links
        self.link_attrs = link_attrs
        self.node_attrs = node_attrs

        self.network_attrs = {}
        if net_attrs: self.network_attrs = net_attrs
        
        # This will add attributes as a column in the links dataframe
        self.put_attributes_in_links()
        self.add_free_travel_time()        

    def only_cars(self):
        """Filter the network to only include links that allow car traffic."""
        if 'modes' in self.links.columns:
            links_to_keep  = self.links['modes'].str.contains('car', na=False)
            links_to_keep &= (~self.links['link_id'].str.contains('pt', na=False))
            self.links = self.links[links_to_keep].reset_index(drop=True)

            nodes_to_keep = set(self.links['from_node']).union(set(self.links['to_node']))
            self.nodes = self.nodes[self.nodes['node_id'].isin(nodes_to_keep)].reset_index(drop=True)
            
            logger.info(f"Filtered network to only keep cars links: {len(self.links)} car links and {len(self.nodes)} nodes.")
        else:
            logger.warning("No 'modes' column found in links. Cannot filter for cars.")

    def add_free_travel_time(self):
        """Add a 'travel_time' column to links based on freespeed and length."""
        if 'freespeed' in self.links.columns and 'length' in self.links.columns:
            self.links['free_travel_time'] = self.links['length'] / self.links['freespeed']
            logger.info("Added travel_time to links based on freespeed and length.")
        else:
            logger.warning("Links do not have 'freespeed' or 'length' columns. Cannot calculate travel time.")

    def put_attributes_in_links(self):
        link_attrs = self.link_attrs.groupby('link_id').apply(lambda x: dict(zip(x['name'], x['value']))).reset_index(name='attributes')
        self.links = self.links.merge(link_attrs, on="link_id",how="left")
        self.links.loc[self.links["attributes"].isna(), "attributes"] = None
        
    def __len__(self):
        return len(self.links)
    
    def __str__(self):
        return 'Network: {nodes} nodes, {links} links, {crs}'.format(
            nodes=len(self.nodes),
            links=len(self.links),
            crs=Network._crsTag in self.network_attrs and self.network_attrs[Network._crsTag] or 'No CRS')
    
    def as_geo(self, projection=None):
        """Return a GeoPandas GeoDataFrame containing link geometries suitable for plotting."""

        logger.info("Converting network to GeoDataFrame")
        
        # Project the coords, if CRS is specified somehow
        if projection:
            crs = {'init': projection}
        elif Network._crsTag in self.network_attrs:
            crs = self.network_attrs[Network._crsTag]
        else:
            crs = None
    
        # attach xy to links
        full_net = (self.links
        .merge(self.nodes,
                left_on='from_node',
                right_on='node_id')
        .merge(self.nodes,
                left_on='to_node',
                right_on='node_id',
                suffixes=('_from_node', '_to_node'))
        )
    
        # create the geometry column from coordinates
        geometry = [shp.LineString([(ox,oy), (dx,dy)]) for ox, oy, dx, dy in zip(full_net.x_from_node, full_net.y_from_node, full_net.x_to_node, full_net.y_to_node)]
    
        # build the geopandas geodataframe
        geo_net = (gpd.GeoDataFrame(full_net,
            geometry=geometry,
            crs = crs)
            .drop(columns=['x_from_node','y_from_node','node_id_from_node','node_id_to_node','x_to_node','y_to_node'])
            )
    
        return geo_net
    
    def as_networkx(self):                                      
        """
        Converts the network to a directed NetworkX graph.

        Returns:
            networkx.DiGraph: Directed graph with edge attributes 'idx', 'link_id',
                            and node attributes 'x', 'y' (coordinates).
        """
        logger.info("Converting network to directed networkx graph")
        links = self.links
        nodes = self.nodes        
        G = nx.DiGraph()

        # Add nodes with coordinates        
        G.add_nodes_from(zip(nodes['node_id'], 
                             ({'x':x,'y':y} for x,y in zip(nodes['x'], nodes['y']))
                             ))

        # Add edges with attributes
        G.add_edges_from(zip(links['from_node'],links['to_node'],
                             ({'link_id': lid, 'free_travel_time':tt} for lid,tt in zip(links['link_id'], links['free_travel_time']))
                          ))
        return G
       
    def as_dual_networkx(self, mode="car"):
        """
        Converts the MATSim network into a dual directed NetworkX graph.
        Nodes in the dual graph represent links in the MATSim network.
        Edges represent allowed transitions between links (i.e., valid turns).
        
        Parameters:
            mode (str): The mode of transport to filter links (default is "car").
        
        Returns:
            networkx.DiGraph: Dual graph with link_ids as nodes and valid transitions as edges.
        """
        logger.info("Converting MATSim network to dual NetworkX graph")

        df = self.links.copy()
        df = df[df['modes'].str.contains(mode, na=False)].reset_index(drop=True)

        # Map from from_node to outgoing link_ids
        links_from_node = df.groupby('from_node')['link_id'].apply(list).to_dict()
        link_info = df.set_index('link_id')

        G = nx.DiGraph()

        for link_id, row in link_info.iterrows():
            to_node = row['to_node']
            
            # Parse disallowed next links safely
            disallowed = set()
            disallowed_links = row.attributes.get("disallowedNextLinks")
            if disallowed_links:
                try:
                    disallowed_dict = eval(disallowed_links)
                    if mode in disallowed_dict:
                        disallowed.update(link for group in disallowed_dict[mode] for link in group)
                except Exception as e:
                    logger.warning(f"Failed to parse disallowedNextLinks for link {link_id}: {e}")

            # All links that start where this link ends (i.e., potential next steps)
            next_links = links_from_node.get(to_node, [])

            for next_link_id in next_links:
                if next_link_id not in disallowed:
                    G.add_edge(link_id, next_link_id)

        return G

    def as_igraph(self):
        """
        Converts the network to an igraph Graph.
        
        Returns:
            igraph.Graph: Graph with edge attributes ('link_id', 'free_travel_time') and
                          vertex attributes ('node_id', 'x', 'y').
        """
        logger.info("Converting network to igraph graph")
    
        # Assign unique indices to nodes and links
        self.nodes["node_idx"] = range(len(self.nodes))
        self.links["link_idx"] = range(len(self.links))
    
        # Map node_id to igraph index
        node_id_to_idx = dict(zip(self.nodes["node_id"], self.nodes["node_idx"]))
    
        # Create directed graph with N vertices
        G = ig.Graph(n=len(self.nodes), directed=True)
    
        # Set vertex attributes
        G.vs["node_id"] = self.nodes["node_id"].tolist()
        G.vs["x"] = self.nodes["x"].tolist()
        G.vs["y"] = self.nodes["y"].tolist()
        #G.vs["node_idx"] = self.nodes["node_idx"].tolist()
    
        # Build edge list using mapped node indices
        edge_tuples = [
            (node_id_to_idx[from_node], node_id_to_idx[to_node])
            for from_node, to_node in zip(self.links["from_node"], self.links["to_node"])
        ]
    
        G.add_edges(edge_tuples)
    
        # Set edge attributes
        G.es["link_id"] = self.links["link_id"].tolist()
        G.es["free_travel_time"] = self.links["free_travel_time"].tolist()
    
        return G
    










def read_network(filename, skip_attributes=False):
    """Read a MATSim network.xml.gz file. Returns a Network object with dataframes
    for nodes, links, node_attributes, and link_attributes. If the network has a CRS
    projection set, it will be available in network_attrs."""
    tree = ET.iterparse(xopen.xopen(filename, 'r'), events=['start', 'end'])
    nodes = []
    links = []
    node_attrs = []
    link_attrs = []

    network_attrs = {}

    attributes = node_attrs
    attr_label = 'node_id'
    current_id = None

    for xml_event, elem in tree:
        # the nodes element CLOSES at the end of the nodes, followed by links:
        if elem.tag == 'links' and xml_event == 'start':
            attributes = link_attrs
            attr_label = 'link_id'

        elif elem.tag == 'node' and xml_event == 'start':
            atts = elem.attrib
            current_id = atts['id']

            atts['node_id'] = atts.pop('id')
            atts['x'] = float(atts['x'])
            atts['y'] = float(atts['y'])
            if 'z' in atts: atts['z'] = float(atts['z'])

            nodes.append(atts)

        elif elem.tag == 'link' and xml_event == 'start':
            atts = elem.attrib
            current_id = atts['id']

            atts['link_id'] = atts.pop('id')
            atts['from_node'] = atts.pop('from')
            atts['to_node'] = atts.pop('to')

            atts['length'] = float(atts['length'])
            atts['freespeed'] = float(atts['freespeed'])
            atts['capacity'] = float(atts['capacity'])
            atts['permlanes'] = float(atts['permlanes'])

            if 'volume' in atts: atts['volume'] = float(atts['volume'])

            links.append(atts)


        elif elem.tag == 'attribute' and xml_event == 'end':
            if elem.attrib['name'] == Network._crsTag:
                network_attrs[Network._crsTag] = elem.text

            elif not skip_attributes:
                atts = {}
                atts[attr_label] = current_id
                atts['name'] = elem.attrib['name']
                atts['value'] = elem.text

                # TODO: pandas will make the value column "object" since we're mixing types
                if 'class' in elem.attrib:
                    if elem.attrib['class'] == 'java.lang.Long':
                        atts['value'] = int(elem.text)
                    if elem.attrib['class'] == 'java.lang.Double':
                        atts['value'] = float(elem.text)
                    if elem.attrib['class'] == 'java.lang.Integer':
                        atts['value'] = int(elem.text)

                attributes.append(atts)

        # clear the element when we're done, to keep memory usage low
        if elem.tag in ['node', 'link'] and xml_event == 'end':
            elem.clear()

    nodes = pd.DataFrame.from_records(nodes)
    links = pd.DataFrame.from_records(links)
    node_attrs = pd.DataFrame.from_records(node_attrs)
    link_attrs = pd.DataFrame.from_records(link_attrs)
    
    # make sure all ids are str
    nodes["node_id"] = nodes["node_id"].astype(str)
    links["link_id"] = links["link_id"].astype(str)
    link_attrs["link_id"] = link_attrs["link_id"].astype(str)
    
    return Network(nodes, links, node_attrs, link_attrs, network_attrs)






import lxml.etree as ETF
def fast_network_reader(filename, skip_attributes=False):
    """Fast reader for MATSim network.xml.gz file."""

    nodes = []
    links = []
    node_attrs = []
    link_attrs = []

    network_attrs = {}

    attr_list = node_attrs
    attr_label = 'node_id'
    current_id = None

    context = ETF.iterparse(xopen.xopen(filename, 'rb'), events=('start', 'end'))

    for event, elem in context:
        tag = elem.tag

        if tag == 'links' and event == 'start':
            attr_list = link_attrs
            attr_label = 'link_id'

        elif tag == 'node' and event == 'start':
            attrib = elem.attrib
            current_id = node_id = attrib['id']
            nodes.append({
                'node_id': node_id,
                'x': float(attrib['x']),
                'y': float(attrib['y']),
                'z': float(attrib['z']) if 'z' in attrib else None,
            })

        elif tag == 'link' and event == 'start':
            attrib = elem.attrib
            current_id = link_id = attrib['id']
            links.append({
                'link_id': link_id,
                'from_node': attrib['from'],
                'to_node': attrib['to'],
                'length': float(attrib['length']),
                'freespeed': float(attrib['freespeed']),
                'capacity': float(attrib['capacity']),
                'permlanes': float(attrib['permlanes']),
                'volume': float(attrib['volume']) if 'volume' in attrib else None,
            })

        elif tag == 'attribute' and event == 'end':
            name = elem.attrib['name']
            if name == Network._crsTag:
                network_attrs[Network._crsTag] = elem.text
            elif not skip_attributes:
                value = elem.text
                cls = elem.attrib.get('class', None)
                if cls == 'java.lang.Long' or cls == 'java.lang.Integer':
                    value = int(value)
                elif cls == 'java.lang.Double':
                    value = float(value)
                attr_list.append({
                    attr_label: current_id,
                    'name': name,
                    'value': value,
                })

        if tag in ('node', 'link') and event == 'end':
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]  # aggressive memory cleanup

    # Convert lists to DataFrames
    nodes_df = pd.DataFrame(nodes)
    links_df = pd.DataFrame(links)
    node_attrs_df = pd.DataFrame(node_attrs)
    link_attrs_df = pd.DataFrame(link_attrs)

    # Ensure correct dtypes
    nodes_df['node_id'] = nodes_df['node_id'].astype(str)
    links_df['link_id'] = links_df['link_id'].astype(str)
    if not node_attrs_df.empty:
        node_attrs_df['node_id'] = node_attrs_df['node_id'].astype(str)
    if not link_attrs_df.empty:
        link_attrs_df['link_id'] = link_attrs_df['link_id'].astype(str)

    return Network(nodes_df, links_df, node_attrs_df, link_attrs_df, network_attrs)







