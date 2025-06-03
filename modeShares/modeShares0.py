#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:39:35 2025

@author: dabdelkader
"""
import os
import pandas as pd
import numpy as np


cache_dir = "/home/dabdelkader/Euler/ch-zh-synpop/cache10p100"

trips_file = "data.microcensus.trips__cc73f02aef57a0f5f7c4e100ba36c3b1.p"
persons_file = "data.microcensus.persons__6adf7cd752243cefe558b1e7261937fa.p"
households_file = "data.microcensus.households__6adf7cd752243cefe558b1e7261937fa.p"


trips, filterout_ids = pd.read_pickle(os.path.join(cache_dir, trips_file))
sel = ~trips["person_id"].isin(filterout_ids)
trips = trips.loc[sel, ['person_id', 'trip_id',
                        'mode', 'crowfly_distance', 'network_distance']]


persons = pd.read_pickle(os.path.join(cache_dir, persons_file))
sel = (~persons["person_id"].isin(filterout_ids)) & (
    persons["weekend"] == False)
persons = persons.loc[sel, ['person_id', 'person_weight',
                            'age', 'sex', 'income_class', 'canton_id']]

# Trips file with weight attributes
trips = trips.merge(persons, how="left", on="person_id")
trips = trips[(trips.household_weight.notna()) & (trips.person_weight.notna())]


################## OVERALL MODE SHARES ##################

# Total weighted trips using person_weight and household_weight
total_person_weight = trips['person_weight'].sum()
total_household_weight = trips['household_weight'].sum()

# Mode shares using person_weight
mode_share_person = (
    trips.groupby('mode')
    .apply(lambda x: x['person_weight'].sum())
    .reset_index(name='mode_share')
)
mode_share_person['mode_share'] = mode_share_person['mode_share'] / \
    total_person_weight
mode_share_person = mode_share_person.set_index("mode")
mode_share_person = mode_share_person.to_dict()["mode_share"]

################## MODE SHARES DISTRIBUTION ##################

# bins for crowfly distance
bins = [0, 1000, 3000, 5000, 8000, 10000, 20000, 1000000]
labels = ['0km-1km', '1km-3km', '3km-5km',
          '5km-8km', '8km-10km', '10km-20km', '20km+']

trips['distance_bin'] = pd.cut(
    trips['crowfly_distance'], bins=bins, labels=labels, include_lowest=True, ordered=True)

# Group by mode and distance bin, and compute weighted shares


mode_distribution_person = (
    trips.groupby(['distance_bin', 'mode'], observed=False)
    .apply(lambda g: g["person_weight"].sum())
    .reset_index(name='mode_share')
)

# Normalize within each distance bin
mode_distribution_person['mode_share'] = mode_distribution_person.groupby('distance_bin', observed=False)['mode_share'].transform(
    lambda x: x / x.sum()
)

mode_distribution_person = mode_distribution_person.sort_values('distance_bin')
mode_distribution_person = mode_distribution_person.groupby('mode') \
                                                   .apply(lambda group: group['mode_share'].tolist()) \
                                                   .to_dict()

output = dict(mode_share = mode_share_person,
              mode_share_distribution = mode_distribution_person,
              distance_bins = bins,
              distances = labels)
