#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:42:45 2025

@author: dabdelkader
"""

from typing import Dict
import os
from MomentumAndDecay.BetaRateRise import BetaRateRise
from MomentumAndDecay.PopulationFactor import PopulationFactor



def parse_dict(input_str: str) -> Dict[str, float]:
    input_str = input_str.replace("\n", "")
    result = {}
    for pair in input_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid key-value pair: {pair}")
        key, value = pair.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            result[key] = float(value)
        except ValueError:
            raise ValueError(f"Invalid float value: {value}")
    return result

def check_required_files(paths: list[str]):
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file does not exist: {path}")


def get_files(args):
    sim_path = args.variables_path
    sim_iter = os.path.basename(sim_path).split('.')[-1]
    
    bike_file = f"{sim_path}/{sim_iter}.choice_variables_bike.csv"
    pt_file = f"{sim_path}/{sim_iter}.choice_variables_pt.csv"
    car_file = f"{sim_path}/{sim_iter}.choice_variables_car.csv"
    walk_file = f"{sim_path}/{sim_iter}.choice_variables_walk.csv"
    cp_file = f"{sim_path}/{sim_iter}.choice_variables_car_passenger.csv"
    tours_file = f"{sim_path}/{sim_iter}.detailed_utilities.csv"
    
    files = dict(bike=bike_file, pt=pt_file, car=car_file, walk=walk_file, car_passenger=cp_file, tours = tours_file)
    
    check_required_files(list(files.values()))
    
    return files


def get_beta_and_population(args, beta_method = "step", population_method = "step"):
    BetaRateRise.set_beta(args.beta_momentum)
    beta = BetaRateRise.get_beta(args.iteration, method=beta_method)

    PopulationFactor.set_population(args.population_sample)
    population = PopulationFactor.get_population(args.iteration, method = population_method)
    
    return beta, population



def get_mode_shares_distribution(args):
    car_mode_share  = list(map(float, args.car_mode_share.split(',')))
    pt_mode_share   = list(map(float, args.pt_mode_share.split(',')))
    walk_mode_share = list(map(float, args.walk_mode_share.split(',')))
    bike_mode_share = list(map(float, args.bike_mode_share.split(',')))
    
    distance_bins = [0.0] + list(map(float, args.distance_bins.split(','))) + [float('inf')]
    
    assert len(car_mode_share)==len(pt_mode_share)==len(walk_mode_share)==len(bike_mode_share), "The length of mode shares must be equal"
    assert len(car_mode_share)==len(distance_bins)-1, "The length of distance bins must be equal to the length of the modes shares + 1."
    
    mode_shares_distribution = dict(distance= distance_bins,car=car_mode_share, pt=pt_mode_share, walk = walk_mode_share, bike = bike_mode_share)
    return mode_shares_distribution









