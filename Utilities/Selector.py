#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:19:01 2025

@author: dabdelkader
"""

from Utilities.TourUtility import TourUtility
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class Selector():
    minimum_utility = -700.0
    maximum_utility = 700.0
    considerMinimumUtility = False
    selector = "MultinomialLogit"  # one of ["MultinomialLogit", "Maximum"]

    @staticmethod
    def get_mode_shares_from_tours(tours: pd.DataFrame):
        # Reduce to only relevant columns early
        utilities = tours[["person_id", "selection_id", "candidate_mode", "utility"]]

        if Selector.selector == "MultinomialLogit":
            return Selector._multinomial_logit_selection(utilities)
        elif Selector.selector == "Maximum":
            return Selector._maximum_utility_selection(utilities)
        else:
            raise ValueError(f"Unknown selector: {Selector.selector}")

    @staticmethod
    def _maximum_utility_selection(df):        

        if Selector.considerMinimumUtility:
            df = df[df['utility'] > Selector.minimum_utility]

        # Group by person_id and selection_id
        grouped = df.groupby(['person_id', 'selection_id'], sort=False)

        # Find the row with max utility per group
        max_rows = df.loc[grouped['utility'].idxmax()].index

        # Create selected column
        df['selected'] = False
        df.loc[max_rows, 'selected'] = True

        return df

    @staticmethod
    def _multinomial_logit_selection(df):        
        result = []

        if Selector.considerMinimumUtility:
            df = df[df['utility'] > Selector.minimum_utility]

        # Group by person_id and selection_id
        grouped = df.groupby(['person_id', 'selection_id'], sort=False)

        for name, group in grouped:
            utilities = np.minimum(group['utility'].values, Selector.maximum_utility)
            exp_utilities = np.exp(utilities)
            probabilities = exp_utilities / exp_utilities.sum()

            chosen_index = np.random.choice(group.index, p=probabilities)
            selected = pd.Series(False, index=group.index)
            selected[chosen_index] = True
            result.append(group.assign(selected=selected.values))

        return pd.concat(result)
    
    @staticmethod
    def set_selector(selector:str):
        Selector.selector = selector
    
    @staticmethod
    def get_selector(selector:str):
        return Selector.selector