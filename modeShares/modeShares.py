#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 09:59:30 2025

@author: dabdelkader
"""
import os
import json
import glob
import hashlib
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import modeShares.utils as U


class ModeShares:
    def __init__(self, eqasim_cache_dir: str, cache_dir: str = "./calibrationCache", overwrite: bool = False):
        self.eqasim_cache_dir = eqasim_cache_dir
        self.cache_dir = cache_dir
        self.overwrite = overwrite

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize all instance variables
        self.trips = None
        self.transit = None
        self.mode_shares = {}
        self.distance_labels = []
        self.distance_bins = []

        # Generate cache file name using deterministic hash
        hash_key = hashlib.sha256(self.eqasim_cache_dir.encode()).hexdigest()
        self.cache_file = os.path.join(self.cache_dir, f"modeShare_{hash_key}.json")

        # Try loading from cache
        if os.path.exists(self.cache_file) and not self.overwrite:
            self.load_from_cache()
        else:
            self._setup_files()
            self.trips, self.transit = self._load_data()
            
            self.distance_labels = ['0-(0km-1km)', '1-(1km-2km)', '2-(2km-3km)',
                                    '3-(3km-4km)','4-(4km-5km)', '5-(5km-8km)',
                                    '6-(8km-12km)','7-(12km-20km)','8-(20km+)'] #)it should be like that for sorting later
            self.distance_bins = [0, 1000, 2000, 3000, 4000, 5000, 8000, 12000, 20000, 1000000]
            
            self.trips['distance_bin'] = pd.cut(self.trips['crowfly_distance'], 
                                                bins=self.distance_bins, 
                                                labels=self.distance_labels, 
                                                include_lowest=True, 
                                                ordered=True)
            
            self.mode_shares["global"] = self._get_mode_shares()
            self.mode_shares["distance"] = self._get_mode_shares_by("distance_bin") #self._get_mode_shares_distribution()
            self.mode_shares["canton"] = self._get_mode_shares_by("canton_id")
            self.mode_shares["income"] = self._get_mode_shares_by("income_class")
            self.mode_shares["age"]    = self._get_mode_shares_by("age_class")
            self.mode_shares["sp_region"]    = self._get_mode_shares_by("sp_region")
            
            self.mode_shares["mode_distance"] = self._get_mode_distribution_by("distance_bin")
            self.mode_shares["mode_canton"] = self._get_mode_distribution_by("canton_id")
            self.mode_shares["mode_income"] = self._get_mode_distribution_by("income_class")
            self.mode_shares["mode_age"]    = self._get_mode_distribution_by("age_class")                        
            
            self.mode_shares["transit"] = self._get_transit_distributions()
            
            self.save_to_cache()
        
        assert len(self.distance_bins)==len(self.distance_labels)+1, "Incorrect distance bins/labels"
        assert len(self.mode_shares["distance"]["car"])==len(self.mode_shares["mode_distance"]["car"])==len(self.distance_labels)
        
    def _setup_files(self):
        """Automatically find latest trips and persons files."""
        trips_files = glob.glob(os.path.join(self.eqasim_cache_dir, "**", "*data.microcensus.trips*.p"), recursive=True)
        persons_files = glob.glob(os.path.join(self.eqasim_cache_dir, "**", "*data.microcensus.persons*.p"), recursive=True)
        transit_files = glob.glob(os.path.join(self.eqasim_cache_dir, "**", "*data.microcensus.transit*.p"), recursive=True)
        if not trips_files or not persons_files or not transit_files:
            raise FileNotFoundError("Could not find required trips/persons/transit files")

        self.trips_file   = max(trips_files, key=os.path.getctime)
        self.persons_file = max(persons_files, key=os.path.getctime)
        self.transit_file = max(transit_files, key=os.path.getctime)

    def _load_data(self):
        trips, filterout_ids = pd.read_pickle(self.trips_file)
        sel = ~trips["person_id"].isin(filterout_ids)
        trips = trips.loc[sel, ['person_id', 'trip_id',
                                'mode', 'crowfly_distance', 'network_distance']]

        persons = pd.read_pickle(self.persons_file)
        sel = (~persons["person_id"].isin(filterout_ids)) & (persons["weekend"] == False)
        persons = persons.loc[sel, ['person_id', 'person_weight', 'age', 'age_class', 'sp_region',
                                    'sex', 'income_class', 'canton_id', 'household_weight']]

        # Merge with persons to get weights
        trips = trips.merge(persons, how="left", on="person_id")
        
        sel = ((trips.household_weight.notna()) & 
               (trips.person_weight.notna()) &
               (trips.crowfly_distance>1) )
        trips = trips[sel].reset_index(drop=True)       
        trips["canton_id"] = trips["canton_id"].astype(int)
        trips["income_class"] = trips["income_class"].astype(int)
        trips["age_class"] = trips["age_class"].astype(int)
        trips["sp_region"] = trips["sp_region"].astype(int)
        
        # loadt trasit
        transit = pd.read_pickle(self.transit_file)
        transit = transit[["person_id","trip_id","in_vehicle_time","line_switches","access_egress_time","waiting_time"]]
        sel = transit.person_id.isin(trips.person_id.unique())
        transit = transit[sel].reset_index(drop=True)
        
        return trips, transit

    def _get_mode_shares(self):
        total_person_weight = self.trips['person_weight'].sum()
        mode_share_person = (
            self.trips.groupby('mode')
            .apply(lambda x: x['person_weight'].sum())
            .reset_index(name='mode_share')
        )
        mode_share_person['mode_share'] /= total_person_weight
        mode_share_person = mode_share_person.set_index("mode").to_dict()["mode_share"]
        mode_share_person = {k:[v,] for k,v in mode_share_person.items()}
        #I transform them into lists so that everything is consistent
        return mode_share_person

    def _get_mode_shares_by(self, by = "canton_id"):
        mode_share =  (self.trips
                        .groupby([by, "mode"], observed=False)["person_weight"]
                        .sum()
                        .groupby(level=by)
                        .transform(lambda x: x / x.sum())                
                        .rename("mode_share")
                        .reset_index()                
                        .pivot(index=by, columns="mode", values="mode_share")
                        .fillna(0)
                        .sort_values(by=by))
        
        mode_share = {mode:mode_share[mode].tolist() for mode in mode_share.columns}            
        return mode_share
    
    def _get_mode_distribution_by(self, by = "distance_bin"):
        mode_share =  (self.trips
                        .groupby([by, "mode"], observed=False)["person_weight"]
                        .sum()
                        .groupby(level="mode")
                        .transform(lambda x: x / x.sum())                
                        .rename("distribution")
                        .reset_index()                
                        .pivot(index=by, columns="mode", values="distribution")
                        .fillna(0)
                        .sort_values(by=by))
        
        mode_share = {mode:mode_share[mode].tolist() for mode in mode_share.columns}            
        return mode_share
                    
    
    def _get_transit_distributions(self):
        cols = ["person_id","trip_id","distance_bin","person_weight"]
        sel  = self.trips["mode"]=="pt"
        df = self.transit.merge(self.trips.loc[sel,cols], on=["person_id","trip_id"], how="left")
        def get_dist_of(x):
            return (df.groupby("distance_bin", observed=False)[[x,"person_weight"]]
                      .apply(lambda g: U.clean_and_weighted_avg(g,x,"person_weight"))
                      .reset_index(name=x)                 
                      .sort_values("distance_bin")[x]
                      .tolist()
                      )
        
        return dict(
            in_vehicle_time    = get_dist_of("in_vehicle_time"),
            line_switches      = get_dist_of("line_switches"),
            waiting_time       = get_dist_of("waiting_time"),
            access_egress_time = get_dist_of("access_egress_time")
            )        
        
        
    def save_to_cache(self):
        outputs = self.get_all_results()
        with open(self.cache_file, "w") as f:
            json.dump(outputs, f)

    def load_from_cache(self):
        with open(self.cache_file, "r") as f:
            outputs = json.load(f)

        self.mode_shares = outputs["mode_shares"]
        self.distance_labels = outputs["distances"]
        self.distance_bins = outputs["distance_bins"]
        

    def get_all_results(self):
        return {
            "mode_shares": self.mode_shares,
            "distance_bins": self.distance_bins,
            "distances": self.distance_labels,
        }
    
    def get_distance_bins(self):
        return self.distance_bins
    
    def get_distance_labels(self):
        return self.distance_labels
    
    def get_mode_shares(self):
        return self.mode_shares
    
    def get_mode_share_by(self, by):
        return self.mode_shares[by]

    
    def plot_mode_share_by_canton(self):
        df = pd.DataFrame(self.mode_shares_by_canton).T

        # Transpose for easier plotting by mode
        modes = df.columns.tolist()
        cantons = df.index.tolist()
        
        ax = df.plot(kind='bar', stacked=True, figsize=(13, 6), cmap='tab20', width=0.8)
                
        plt.xlabel('Canton ID', fontsize=14)
        plt.ylabel('Mode Share (%)', fontsize=14)
        plt.xticks(rotation=0)
        plt.legend(title='Mode', fontsize=14, ncols=5,  loc='lower center',      
                   bbox_to_anchor=(0.5, 0.87), 
                   bbox_transform=plt.gcf().transFigure)
        plt.tight_layout()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)  
        for i, rect in enumerate(ax.patches):
            width = rect.get_width()
            height = rect.get_height()
            x = rect.get_x()
            y = rect.get_y()
            
            label_text = f'{height:.1%}'
            
            if height > 0.05:  # only label if large enough
                ax.text(x + width/2, y + height/2, label_text,
                        ha='center', va='center', fontsize=8, color='white')
        
        plt.show()
        
    def plot_mode_share_distribution(self):
        df = pd.DataFrame(self.mode_shares_distribution)
        df.index = self.distance_labels
        
        # Transpose for easier plotting by mode
        modes = df.columns.tolist()
        distances = df.index.tolist()
        
        ax = df.plot(kind='bar', stacked=True, figsize=(13, 6), cmap='tab20', width=0.8)
        
        plt.xlabel('Distance bin', fontsize=14)
        plt.ylabel('Mode Share (%)', fontsize=14)
        plt.xticks(rotation=0)
        plt.legend(title='Mode', fontsize=14, ncols=5,  loc='lower center',      
                   bbox_to_anchor=(0.5, 0.87), 
                   bbox_transform=plt.gcf().transFigure)
        plt.tight_layout()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)   
        for i, rect in enumerate(ax.patches):
            width = rect.get_width()
            height = rect.get_height()
            x = rect.get_x()
            y = rect.get_y()
            
            label_text = f'{height:.1%}'
            
            if height > 0.05:  # only label if large enough
                ax.text(x + width/2, y + height/2, label_text,
                        ha='center', va='center', fontsize=12, color='white')
        
        plt.show()