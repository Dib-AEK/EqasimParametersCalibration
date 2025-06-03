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


class ModeShares:
    def __init__(self, eqasim_cache_dir: str, cache_dir: str = "./calibrationCache", overwrite: bool = False):
        self.eqasim_cache_dir = eqasim_cache_dir
        self.cache_dir = cache_dir
        self.overwrite = overwrite

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize all instance variables
        self.trips = None
        self.mode_shares = {}
        self.distance_labels = []
        self.distance_bins = []
        self.mode_share_distribution = {}

        # Generate cache file name using deterministic hash
        hash_key = hashlib.sha256(self.eqasim_cache_dir.encode()).hexdigest()
        self.cache_file = os.path.join(self.cache_dir, f"modeShare_{hash_key}.json")

        # Try loading from cache
        if os.path.exists(self.cache_file) and not self.overwrite:
            self.load_from_cache()
        else:
            self._setup_files()
            self.trips = self._load_data()
            self.mode_shares = self._get_mode_shares()
            self.distance_labels = ['0km-1km', '1km-3km', '3km-5km', '5km-8km', '8km-10km', '10km-20km', '20km+']
            self.distance_bins = [0, 1000, 3000, 5000, 8000, 10000, 20000, 1000000]
            self.mode_shares_distribution = self._get_mode_shares_distribution()
            self.mode_shares_by_canton = self._get_mode_shares_by_canton()
            self.save_to_cache()

    def _setup_files(self):
        """Automatically find latest trips and persons files."""
        trips_files = glob.glob(os.path.join(self.eqasim_cache_dir, "**", "*data.microcensus.trips*.p"), recursive=True)
        persons_files = glob.glob(os.path.join(self.eqasim_cache_dir, "**", "*data.microcensus.persons*.p"), recursive=True)

        if not trips_files or not persons_files:
            raise FileNotFoundError("Could not find required trips/persons files")

        self.trips_file = max(trips_files, key=os.path.getctime)
        self.persons_file = max(persons_files, key=os.path.getctime)

    def _load_data(self):
        trips, filterout_ids = pd.read_pickle(self.trips_file)
        sel = ~trips["person_id"].isin(filterout_ids)
        trips = trips.loc[sel, ['person_id', 'trip_id',
                                'mode', 'crowfly_distance', 'network_distance']]

        persons = pd.read_pickle(self.persons_file)
        sel = (~persons["person_id"].isin(filterout_ids)) & (persons["weekend"] == False)
        persons = persons.loc[sel, ['person_id', 'person_weight',
                                    'age', 'sex', 'income_class', 'canton_id', 'household_weight']]

        # Merge with persons to get weights
        trips = trips.merge(persons, how="left", on="person_id")
        
        sel = ((trips.household_weight.notna()) & 
               (trips.person_weight.notna()) &
               (trips.crowfly_distance>1) )
        trips = trips[sel]        
        trips["canton_id"] = trips["canton_id"].astype(int)
        return trips

    def _get_mode_shares(self):
        total_person_weight = self.trips['person_weight'].sum()
        mode_share_person = (
            self.trips.groupby('mode')
            .apply(lambda x: x['person_weight'].sum())
            .reset_index(name='mode_share')
        )
        mode_share_person['mode_share'] /= total_person_weight
        return mode_share_person.set_index("mode").to_dict()["mode_share"]

    def _get_mode_shares_distribution(self):
        bins = self.distance_bins
        labels = self.distance_labels

        self.trips['distance_bin'] = pd.cut(
            self.trips['crowfly_distance'], bins=bins, labels=labels, include_lowest=True, ordered=True
        )

        mode_distribution_person = (
            self.trips.groupby(['distance_bin', 'mode'], observed=False)
            .apply(lambda g: g["person_weight"].sum())
            .reset_index(name='mode_share')
        )

        mode_distribution_person['mode_share'] = mode_distribution_person.groupby('distance_bin', observed=False)['mode_share'].transform(
            lambda x: x / x.sum()
        )

        mode_distribution_person = mode_distribution_person.sort_values('distance_bin')

        return mode_distribution_person.groupby('mode') \
                                       .apply(lambda group: group['mode_share'].tolist()) \
                                       .to_dict()

    def _get_mode_shares_by_canton(self):
        return (self.trips.groupby(["canton_id", "mode"], observed=False)["person_weight"]
                .sum()
                .groupby(level="canton_id")
                .transform(lambda x: x / x.sum())
                .rename("mode_share")
                .reset_index()                
                .pivot(index="canton_id", columns="mode", values="mode_share")
                .fillna(0)
                .to_dict(orient="index"))

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
        self.mode_shares_distribution = outputs["mode_shares_distribution"]
        self.mode_shares_by_canton = outputs["mode_shares_by_canton"]

    def get_all_results(self):
        return {
            "mode_shares": self.mode_shares,
            "mode_shares_distribution": self.mode_shares_distribution,
            "distance_bins": self.distance_bins,
            "distances": self.distance_labels,
            "mode_shares_by_canton": self.mode_shares_by_canton
        }
    
    def get_distance_bins(self):
        return self.distance_bins
    
    def get_distance_labels(self):
        return self.distance_labels
    
    def get_mode_shares(self):
        return self.mode_shares
    
    def get_mode_shares_distribution(self):
        return self.mode_shares_distribution    
    
    def get_mode_share_by_canton(self):
        return self.mode_shares_by_canton
    
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