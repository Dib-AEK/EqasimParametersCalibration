#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:22:46 2025

@author: dabdelkader
"""

from Utilities.BaseUtility import BaseUtility
from Utilities.BikeUtility import BikeUtility
from Utilities.CarUtility import CarUtility
from Utilities.PtUtility import PtUtility
from Utilities.WalkUtility import WalkUtility
from Utilities.ZeroUtility import ZeroUtility
import pandas as pd
import numpy as np
from scipy.stats import qmc
import polars as pl

class TourUtility(BaseUtility):
    utility_estimators = {
        "car": CarUtility,
        "pt": PtUtility,
        "walk": WalkUtility,
        "bike": BikeUtility,
        "car_passenger": ZeroUtility
    }

    # Shared class-level variables for dataframes
    variables_by_mode = {}
    tours = None
    persons = []
    num_persons = 0
    sample = None
    sobol_generator = qmc.Sobol(d=1, scramble=True)
    use_sobol = False
    
    @staticmethod
    def init_data(car, pt, bike, walk, cp, tours=None, population_sample = None):
        """
        Initializes mode-specific input data once.
        """
        TourUtility.variables_by_mode = {
            "car": car,
            "pt": pt,
            "bike": bike,
            "walk": walk,
            "car_passenger":cp
        }
        
        TourUtility.sample = population_sample
        
        if tours is not None:
            TourUtility.tours = tours
            TourUtility.persons = tours["person_id"].unique()
            TourUtility.num_persons = len(TourUtility.persons)
            # Here we include Euclidean distance in the tours dataframe in order to get mode shares distribution
            TourUtility.create_distance_column_in_tours()
      
    @staticmethod
    def set_population_sample(population_sample):
        TourUtility.sample = population_sample
            
    @staticmethod
    def get_utility_of(person_id, trip_index, mode):
        if mode=="car_passenger":
            return 0.0
        estimator = TourUtility.utility_estimators[mode]        
        variables = TourUtility.variables_by_mode[mode].loc[f"{person_id}_{trip_index}"]
        return estimator.compute(variables)

    @staticmethod
    def compute(tour):
        """
        Computes utilities for each trip in the tour using pre-initialized data.

        Parameters:
        - tour: an object with attributes `person_id`, `trips_index`, and `candidate_mode`

        Returns:
        - List[float]: computed utilities
        """
        if not TourUtility.variables_by_mode:
            raise RuntimeError("TourUtility data not initialized. Call init_data() first.")

        person_id = tour.person_id
        trips = zip(tour.trips_index, tour.candidate_mode)

        return [TourUtility.get_utility_of(person_id, trip_index, mode) for trip_index, mode in trips]
    
    @staticmethod
    def get_all_utilities_unifficient():
        if TourUtility.tours is None:
            raise RuntimeError("Tours are not in the variables.")
            
        tours = TourUtility.tours.copy()
        
        tours["utility"] = tours.apply(lambda row: sum(TourUtility.compute(row)), axis=1)        
        return tours[["person_id", "selection_id", "trips_index", "candidate_mode", "utility"]]
    
    @staticmethod
    def get_population_sample():
        if TourUtility.use_sobol:
            samples = TourUtility.sobol_generator.random(TourUtility.sample)
            indices = np.floor(samples.flatten() * TourUtility.num_persons).astype(int)
            unique_indices = np.unique(indices)             
            sample_population = TourUtility.persons[unique_indices]
        else:
            sample_population = np.random.choice(TourUtility.persons, size=TourUtility.sample)  
        
        return sample_population
    
    @staticmethod
    def get_all_utilities():
        if TourUtility.tours is None:
            raise RuntimeError("Tours are not initialized.")
        
        cols = ['person_id', 'trip_key', 'selection_id', 'candidate_mode','euclidean_distance']
        tours = TourUtility.tours.select(cols)
        
        # Only select a sample
        if TourUtility.sample is not None and TourUtility.sample < TourUtility.num_persons:
            sample_population = TourUtility.get_population_sample()
            tours = tours.filter(pl.col("person_id").is_in(sample_population)).with_row_index(name="tour_row_id")
        else:
            tours = tours.with_row_index(name="tour_row_id")
        
        
        # Explode tours into individual trips
        exploded = (tours.select(['trip_key','candidate_mode'])
                    .explode(['trip_key', 'candidate_mode'])        
                    .with_columns(pl.lit(0.0).alias("utility")))
        

        # Process each mode's trips in a vectorized manner
        for mode, estimator in TourUtility.utility_estimators.items():
            if mode == "car_passenger":
                continue
    
            variables_df = TourUtility.variables_by_mode.get(mode)
            if variables_df is None:
                raise RuntimeError(f"Missing variables dataframe for mode {mode}.")
            
            mode_mask = exploded["candidate_mode"] == mode
            mode_trips = exploded.filter(mode_mask)
            if mode_trips.is_empty():
                continue
            
            trip_keys = mode_trips["trip_key"].to_list()
            try:
                mode_vars = variables_df.filter(pl.col("trip_key").is_in(trip_keys))
            except Exception as e:
                raise RuntimeError(f"Missing keys for mode {mode}: {e}")
            
            if not mode_vars.is_empty():
                utilities = estimator.compute(mode_vars)
                utility_series = pl.Series("utility", utilities)
                
                mode_update = mode_trips.select("trip_key", "tour_row_id").with_columns(utility_series)
                exploded = exploded.join(mode_update, on=["tour_row_id", "trip_key"], how="left").with_columns(
                    pl.coalesce([pl.col("utility_right"), pl.col("utility")]).alias("utility")
                ).drop("utility_right")
                
                
        
        # Aggregate utilities per original tour
        aggregated = exploded.group_by("tour_row_id").agg(pl.col("utility").sum().alias("utility"))
        
        # Join utilities back to original tours
        updated_tours = tours.join(aggregated, on="tour_row_id").drop("tour_row_id")
        
        return updated_tours.select([*cols, "utility"])


    @staticmethod
    def read_csv(file_path):
        df = (
            pl.read_csv(file_path, separator=";")
            .with_columns(
                # Split strings into lists
                pl.col("trips_index").str.split(","),
                pl.col("candidate_mode").str.split(","),
                
                # Split utilities and cast to float list
                pl.col("utilities").str.split(",")
                .list.eval(pl.element().cast(pl.Float32))
            )
            .rename({
                "utilities": "eqasim_utilities",
                "utility": "eqasim_utility",
                "selected": "eqasim_selected"
            })
        )
        return df
    
    @staticmethod
    def create_distance_column_in_tours():
        # Add a stable row index to preserve original tour rows
        tours = TourUtility.tours.with_row_index(name="tour_row_id").select(
                ['tour_row_id', 'person_id', 'trips_index', 'candidate_mode'])
    
        # Explode tours into individual trips
        exploded = (tours.explode(["trips_index", "candidate_mode"])
                    .with_columns(
                        (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trips_index").cast(pl.Utf8))
                        .alias("trip_key"))
                    .with_columns(pl.lit(None).cast(pl.Float64).alias("euclidean_distance"))
                    )
    
        for mode in ["car", "pt", "walk", "bike", "car_passenger"]:
            variables_df = TourUtility.variables_by_mode.get(mode)
            if variables_df is None:
                raise RuntimeError(f"Missing variables dataframe for mode {mode}.")
    
            variables_df = variables_df.select(["trip_key", "euclideanDistance_km"]
                            ).rename({"euclideanDistance_km": "euclidean_distance"})            
    
            # Join distances on trip_key
            exploded = exploded.join(variables_df, on="trip_key", how="left")
    
            # Update only rows where candidate_mode == mode
            exploded = exploded.with_columns(
                pl.when(pl.col("candidate_mode") == mode)
                  .then(pl.col("euclidean_distance_right"))  # from join
                  .otherwise(pl.col("euclidean_distance"))   # keep existing
                  .alias("euclidean_distance")
            ).drop("euclidean_distance_right")
    
        # Group back by original tour row ID and collect euclidean_distance as list
        updated_tours = (exploded.select(["tour_row_id","trip_key","euclidean_distance"])
                         .group_by("tour_row_id")
                         .agg([pl.col("trip_key"),pl.col("euclidean_distance")])
                         .sort("tour_row_id"))
        
        TourUtility.tours = TourUtility.tours.with_columns([
                updated_tours["euclidean_distance"].alias("euclidean_distance"),
                updated_tours["trip_key"].alias("trip_key")
            ])
            

        
                
    @staticmethod
    def read_and_init(file_path, files:dict, population_sample = None):
        tours = TourUtility.read_csv(file_path)        
        
        
        bike = BikeUtility.read_csv(files["bike"])
        car  = CarUtility.read_csv(files["car"])
        pt   = PtUtility.read_csv(files["pt"])
        walk = WalkUtility.read_csv(files["walk"])                                
        cp   = ZeroUtility.read_csv(files["car_passenger"]) 
        
        TourUtility.init_data(car, pt, bike, walk, cp, tours, population_sample=population_sample)











    

    