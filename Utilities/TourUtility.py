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
    def init_data(car, pt, bike, walk, tours=None, population_sample = None):
        """
        Initializes mode-specific input data once.
        """
        TourUtility.variables_by_mode = {
            "car": car,
            "pt": pt,
            "bike": bike,
            "walk": walk,
            "car_passenger":None
        }
        
        TourUtility.sample = population_sample
        
        if tours is not None:
            TourUtility.tours = tours
            TourUtility.persons = tours.person_id.unique()
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
        
        tours = TourUtility.tours[['person_id', 'trips_index', 'selection_id', 'candidate_mode']].copy()
        
        # Only select a sample
        if TourUtility.sample is not None and TourUtility.sample<TourUtility.num_persons:            
            sample_population = TourUtility.get_population_sample()
            tours = tours[tours.person_id.isin(sample_population)].reset_index(drop=True)
        
        
        # Explode tours into individual trips
        exploded = tours.explode(['trips_index', 'candidate_mode'])
        exploded['trip_key'] = (exploded['person_id'].astype(str) + '_'
                                + exploded['trips_index'].astype(str))
        exploded['utility'] = 0.0  # Initialize utility

        # Process each mode's trips in a vectorized manner
        for mode, estimator in TourUtility.utility_estimators.items():
            if mode == 'car_passenger':
                continue  # Already initialized to 0
            
            variables_df = TourUtility.variables_by_mode.get(mode)
            if variables_df is None:
                raise RuntimeError(f"Missing variables dataframe for mode {mode}.")
            
            mask = exploded['candidate_mode'] == mode
            mode_trips = exploded.loc[mask]
            if mode_trips.empty:
                continue
            
            try:
                # Align variables with mode_trips via index
                mode_vars = variables_df.reindex(mode_trips['trip_key'])
            except KeyError:
                raise RuntimeError(f"Missing keys for mode {mode}.")
            
            if not mode_vars.empty:
                # Vectorized computation (ensure estimator can handle DataFrame)
                utilities = estimator.compute(mode_vars)
                exploded.loc[mask, 'utility'] = utilities.values
        
        # Sum utilities by original tour index
        tours['utility'] = exploded.groupby(level=0)['utility'].sum()
        return tours[["person_id", "selection_id", "trips_index", "candidate_mode", "utility"]]


    @staticmethod
    def read_csv(file_path):
        df = pd.read_csv(file_path, sep=";")
        df["trips_index"] = df["trips_index"].str.split(',')
        df["candidate_mode"] = df["candidate_mode"].str.split(',')
        df["utilities"] = df["utilities"].str.split(',').apply(lambda lst: [float(x) for x in lst])
        df = df.rename(columns={"utilities":"eqasim_utilities",
                                "utility":"eqasim_utility",
                                "selected":"eqasim_selected"})
        return df
    
    
    @staticmethod
    def read_and_init(file_path, files:dict, population_sample = None):
        tours = TourUtility.read_csv(file_path)
        
        bike = BikeUtility.read_csv(files["bike"])
        car  = CarUtility.read_csv(files["car"])
        pt   = PtUtility.read_csv(files["pt"])
        walk = WalkUtility.read_csv(files["walk"])        
    
        TourUtility.init_data(car, pt, bike, walk, tours, population_sample=population_sample)
    











    

    