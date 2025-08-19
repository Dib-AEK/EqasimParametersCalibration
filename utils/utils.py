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
import json
import time
import hashlib
import pandas as pd
import logging
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stable_hash(*args):
    m = hashlib.sha256()
    for arg in args:
        m.update(str(arg).encode())
    return m.hexdigest()


def parse_dict(input_str: str) -> Dict[str, float]:
    input_str = input_str.replace("\n", "")
    result = {}
    for pair in parse_list(input_str):        
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


def parse_list(input_str: str) -> list:
    input_str = input_str.replace("\n", "")
    result = []
    for item in input_str.split(","):
        item = item.strip()
        if item:
            result.append(item)
    return result

def check_if_files_exists(paths: list[str], check_if_files = False):
    for path in paths:
        if not os.path.exists(path):
            raise FileExistsError(f"Path does not exist: {path}")
        if check_if_files:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required file does not exist: {path}")


def get_files(args):
    sim_path = args.variables_path
    if isinstance(sim_path, str):
        sim_iter = os.path.basename(sim_path).split('.')[-1]
        bike_file = os.path.join(sim_path, f"{sim_iter}.choice_variables_bike.csv")
        pt_file = os.path.join(sim_path, f"{sim_iter}.choice_variables_pt.csv")
        car_file = os.path.join(sim_path, f"{sim_iter}.choice_variables_car.csv")
        walk_file = os.path.join(sim_path, f"{sim_iter}.choice_variables_walk.csv")
        cp_file = os.path.join(sim_path, f"{sim_iter}.choice_variables_car_passenger.csv")
        tours_file = os.path.join(sim_path, f"{sim_iter}.detailed_utilities.csv")

        files = {
            "bike": bike_file,
            "pt": pt_file,
            "car": car_file,
            "walk": walk_file,
            "car_passenger": cp_file,
            "tours": tours_file
        }

        check_if_files_exists(list(files.values()), check_if_files=True)
        return files
    else:
        logger.info("Multiple simulation paths provided, concatenating files.")

        all_files = []
        for sim_path_i in sim_path:
            updated_args = args
            updated_args.variables_path = sim_path_i
            all_files.append(get_files(updated_args))

        cache_path = args.optimizer_cache
        concatenated_files = {}

        for key in all_files[0].keys():
            new_file_path = os.path.join(cache_path, f"{key}.csv")
            df_list = []
            person_ids = set()
            for files_dict in all_files:
                df = pl.read_csv(files_dict[key], separator=";")
                # Exclude duplicate person_ids
                df = df.filter(~df["person_id"].is_in(list(person_ids)))
                person_ids.update(df["person_id"].unique().to_list())
                df_list.append(df)
            
            combined_df = pl.concat(df_list, how="vertical")
            combined_df.write_csv(new_file_path, separator=";")

            concatenated_files[key] = new_file_path

        return concatenated_files





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





def hash_run(args):
    cache_dir = args.optimizer_cache
    eqasim_cache = args.eqasim_cache_path
    metric = args.metric
    optimizer = args.optimizer
    objectives = args.objectives
    # create a hash for these data and then a json file path in the cache
    data_hash = stable_hash((*tuple(objectives), metric, optimizer, eqasim_cache, cache_dir))
    return data_hash

def update_number_of_runs(args):
    cache_dir = args.optimizer_cache
    data_hash = hash_run(args)
    json_file_path = os.path.join(cache_dir, f"run_{data_hash}.json")

    now = time.time()
    four_hours = 4 * 3600

    if os.path.exists(json_file_path):
        last_modified = os.path.getmtime(json_file_path)
        if now - last_modified > four_hours:
            data = {"number_of_runs": 1}
        else:
            with open(json_file_path, "r") as f:
                data = json.load(f)
            data["number_of_runs"] += 1
        with open(json_file_path, "w") as f:
            json.dump(data, f)
    else:
        with open(json_file_path, "w") as f:
            json.dump({"number_of_runs": 1}, f)

def get_number_of_runs(args):
    cache_dir = args.optimizer_cache
    data_hash = hash_run(args)
    json_file_path = os.path.join(cache_dir, f"run_{data_hash}.json")

    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        return data.get("number_of_runs", 0)
    return 0
