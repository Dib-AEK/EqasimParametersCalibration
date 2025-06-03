#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:16:33 2025

@author: dabdelkader
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Utilities.BikeUtility import BikeUtility
from Utilities.CarUtility import CarUtility
from Utilities.PtUtility import PtUtility
from Utilities.WalkUtility import WalkUtility
from Utilities.TourUtility import TourUtility


simulation_file = "../sim_output"
bike_files      = [f"{simulation_file}/ITERS/it.{i}/{i}.choice_variables_bike.csv" for i in range(len(os.listdir(simulation_file+"/ITERS")))]
pt_files        = [f"{simulation_file}/ITERS/it.{i}/{i}.choice_variables_pt.csv" for i in range(len(os.listdir(simulation_file+"/ITERS")))]
car_files       = [f"{simulation_file}/ITERS/it.{i}/{i}.choice_variables_car.csv" for i in range(len(os.listdir(simulation_file+"/ITERS")))]
walk_files      = [f"{simulation_file}/ITERS/it.{i}/{i}.choice_variables_walk.csv" for i in range(len(os.listdir(simulation_file+"/ITERS")))]
tours_files     = [f"{simulation_file}/ITERS/it.{i}/{i}.detailed_utilities.csv" for i in range(len(os.listdir(simulation_file+"/ITERS")))]


for i,(bike_file, pt_file, car_file, walk_file, tours_file) in enumerate(
        zip(bike_files, pt_files, car_files, walk_files, tours_files)):
    if i<1:
        continue
    
    bike = BikeUtility.read_csv(bike_file)
    car  = CarUtility.read_csv(car_file)
    pt   = PtUtility.read_csv(pt_file)
    walk = WalkUtility.read_csv(walk_file)
    tours = TourUtility.read_csv(tours_file)
    
    TourUtility.init_data(car, pt, bike, walk)
    
    
    bike["constructed_utility"] = bike.apply(lambda row: BikeUtility.compute(row), axis=1)
    car["constructed_utility"]  = car.apply(lambda row: CarUtility.compute(row), axis=1)
    walk["constructed_utility"] = walk.apply(lambda row: WalkUtility.compute(row), axis=1)
    pt["constructed_utility"]   = pt.apply(lambda row: PtUtility.compute(row), axis=1)
    tours["constructed_utility"] = tours.apply(lambda row: TourUtility.compute(row), axis=1)
    
    if not np.allclose(bike["utility"], bike["constructed_utility"], atol=5e-3):
        print("Constructed and used utilities are not all equal for bike model!")
    
    if not np.allclose(car["utility"], car["constructed_utility"], atol=5e-3):
        print("Constructed and used utilities are not all equal for car model!")
    
    if not np.allclose(pt["utility"], pt["constructed_utility"], atol=5e-3):
        print("Constructed and used utilities are not all equal for pt model!")
    
    if not np.allclose(walk["utility"], walk["constructed_utility"], atol=5e-3):
        print("Constructed and used utilities are not all equal for walk model!")    
    
    
    tours_utilities = tours[["eqasim_utilities","constructed_utility"]].explode(["eqasim_utilities","constructed_utility"])
    if not np.allclose(tours_utilities["eqasim_utilities"].astype(float), 
                       tours_utilities["constructed_utility"].astype(float), 
                       atol=1e-4):
        print("Constructed and used utilities are not all equal for tour model!") 
    
    # Check the efficient implimentation:
    TourUtility.read_and_init(tours_file,{"car" :car_file,
                                          "pt"  :pt_file,
                                          "bike":bike_file,
                                          "walk":walk_file,})
    
    u = TourUtility.get_all_utilities()[["utility"]].rename(columns={"utility":"constructed_utility"})
    u["eqasim_utility"] = TourUtility.tours["eqasim_utility"]
    
    if not np.allclose(u["eqasim_utility"].astype(float), 
                       u["constructed_utility"].astype(float), 
                       atol=1e-4):
        print("Constructed and used utilities are not all equal for efficient tour model!") 
        










