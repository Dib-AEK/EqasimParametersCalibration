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
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import time
import glob
import os


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
    exploded_tours = None
    persons = []
    num_persons = 0
    sample = None
    
    @staticmethod
    def init_data(car, pt, bike, walk, cp, tours=None, population_sample = None, eqasim_cache_dir=None):
        """
        Initializes mode-specific input data once.
        """
        TourUtility.variables_by_mode = {
            "car": car.lazy(),
            "pt": pt.lazy(),
            "bike": bike.lazy(),
            "walk": walk.lazy(),
            "car_passenger":cp.lazy()
        }
        
        TourUtility.sample = population_sample
        TourUtility.eqasim_cache_dir = eqasim_cache_dir
        
        if tours is not None:
            TourUtility.tours = tours.lazy()
            # Here we include Euclidean distance in the tours dataframe in order to get mode shares distribution
            TourUtility.create_distance_column_in_tours()
            # Here, we include other attributes (age, income, sex, canton) for distributions
            if eqasim_cache_dir is not None:
                TourUtility.add_person_attributes_to_tours()
            
            # for better efficiency, we explode tours here
            exploded_tours = TourUtility.get_exploded_tours_for_utilities()
            TourUtility.exploded_tours = {k:v.lazy() for k,v in exploded_tours.items()}
            
            TourUtility.persons = tours["person_id"].unique()
            TourUtility.num_persons = len(TourUtility.persons)
            
            
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
        sample_population = np.random.choice(TourUtility.persons, size=TourUtility.sample)  
        return sample_population
    
    
    @staticmethod
    def get_exploded_tours_for_utilities():
        if TourUtility.tours is None:
            raise RuntimeError("Tours are not initialized.")
    
        cols = ["tour_row_id", "trip_key", "candidate_mode"]

        # Explode trips and candidate modes
        exploded_lazy = (
            TourUtility.tours.select(cols)
            .explode(["trip_key", "candidate_mode"])
            .with_columns([
            pl.col("candidate_mode").cast(pl.Categorical)
            ])
        ).collect()
        
        exploded_lazy = {mode: exploded_lazy.filter(pl.col("candidate_mode") == mode)
                         for mode in TourUtility.utility_estimators}
        return exploded_lazy
    
    @staticmethod
    def compute_mode_utilities(exploded_lazy: pl.LazyFrame, mode: str) -> pl.LazyFrame:
        estimator = TourUtility.utility_estimators.get(mode)
        variables_lazy = TourUtility.variables_by_mode.get(mode)
    
        return (
            exploded_lazy
            .join(variables_lazy, on="trip_key", how="left")
            .with_columns([
                estimator.compute_lazy().cast(pl.Float64)
                .alias("utility")
            ])
            .select(["tour_row_id","utility"])
        )
        
        
    @staticmethod
    def get_all_utilities():
        #select data
        exploded_lazy = TourUtility.exploded_tours
        cols = ['tour_row_id', 'person_id', 'trip_key', 'selection_id', 'candidate_mode', 'euclidean_distance',
                'age_class','sex','income_class','canton_id', 'sp_region']
        tours_lazy = TourUtility.tours.select(cols)
        
        # Compute utilities per mode
        results = (pl.concat([ TourUtility.compute_mode_utilities(exploded_lazy[mode], mode)
                              for mode in TourUtility.utility_estimators])
                   .group_by("tour_row_id")
                   .agg(pl.col("utility").sum().alias("utility")))
                   
        
        # join with tours and return results
        results = (tours_lazy
                    .join(results, on="tour_row_id", how="left")
                    .select([*cols, "utility"]))
        
        return results
        

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
            .with_row_index(name="tour_row_id")
        )
        return df
    
    @staticmethod
    def create_distance_column_in_tours():
        # Add a stable row index to preserve original tour rows
        tours = TourUtility.tours.select(
                ['tour_row_id', 'person_id', 'trips_index', 'candidate_mode']).collect()
    
        # Explode tours into individual trips
        exploded = (tours.explode(["trips_index", "candidate_mode"])
                    .with_columns(
                        (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trips_index").cast(pl.Utf8))
                        .alias("trip_key"))
                    .with_columns(pl.lit(None).cast(pl.Float64).alias("euclidean_distance"))
                    )
    
        for mode in ["car", "pt", "walk", "bike", "car_passenger"]:
            variables_df = TourUtility.variables_by_mode.get(mode).collect()
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
        
        tours = TourUtility.tours.collect().with_columns([
                updated_tours["euclidean_distance"].alias("euclidean_distance"),
                updated_tours["trip_key"].alias("trip_key")
            ])
        
        TourUtility.tours = tours.lazy()
            
    
    @staticmethod
    def add_person_attributes_to_tours(attributes=["age_class","sex","income_class","canton_id", "sp_region"]): 
        eqasim_cache_dir = TourUtility.eqasim_cache_dir
        if eqasim_cache_dir is None:
            return
        
        persons_file = glob.glob(os.path.join(eqasim_cache_dir, "**", "*synthesis.population.enriched*.p"), recursive=True)
        persons_file= max(persons_file, key=os.path.getctime)
        
        
        persons = pd.read_pickle(persons_file)        
        persons = persons.astype({ "age_class": int,
                                   "sex": int,
                                   "income_class": int,
                                   "canton_id": int,
                                   "sp_region":int})

        persons = pl.from_pandas(persons[["person_id",*attributes]])        
        
        tours = TourUtility.tours.collect()
        tours = tours.join(persons, on="person_id", how="left")
        
        assert tours.select(pl.col("sex").is_nan().sum()).item()==0, "Some agents are not found!"
        assert tours.select(pl.col("sp_region").is_nan().sum()).item()==0, "Some spRegions are not found!"
        
        TourUtility.tours = tours.lazy()
        
        
    @staticmethod
    def read_and_init(file_path, files:dict, population_sample = None, eqasim_cache_dir = None):
        tours = TourUtility.read_csv(file_path)        
        
        
        bike = BikeUtility.read_csv(files["bike"])
        car  = CarUtility.read_csv(files["car"])
        pt   = PtUtility.read_csv(files["pt"])
        walk = WalkUtility.read_csv(files["walk"])                                
        cp   = ZeroUtility.read_csv(files["car_passenger"]) 
        
        TourUtility.init_data(car, pt, bike, walk, cp, tours, 
                              population_sample=population_sample,
                              eqasim_cache_dir = eqasim_cache_dir)











    

    