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
from scipy.stats import qmc


class Selector():
    minimum_utility = -700.0
    maximum_utility = 700.0
    considerMinimumUtility = False
    selector = "MultinomialLogit"  # one of ["MultinomialLogit", "Maximum"]
    
    sobol_generator = qmc.Sobol(d=1, scramble=True)
    use_sobol = False
    
    @staticmethod
    def get_mode_shares_from_tours(tours: pd.DataFrame):
        # Reduce to only relevant columns early
        utilities = tours[["person_id", "selection_id", "candidate_mode", "utility"]]
        #TODO: I need to add SOBOL sequence based generator to see if it works better.
        if Selector.selector == "MultinomialLogit":
            if Selector.use_sobol:
                return Selector._multinomial_logit_selection_sobol(utilities)
            else:
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
        if Selector.considerMinimumUtility:
            df = df[df['utility'] > Selector.minimum_utility]
        
        df.loc[:,'utility'] = np.minimum(df['utility'], Selector.maximum_utility)
        
        # Compute the probability
        df.loc[:,"exp_util"] = np.exp(df['utility'])        
        group_sum = df.groupby(['person_id', 'selection_id'])['exp_util'].transform('sum')
        df.loc[:,'probability'] = df["exp_util"] / group_sum
               
        # Generate a random number per row
        df.loc[:,'rand'] = np.random.rand(len(df))
        df.loc[:,'rand'] = df.groupby(['person_id', 'selection_id'])['rand'].transform('first')
        
        # Compute cumulative probability within each group
        df.loc[:,'cum_prob'] = df.groupby(['person_id', 'selection_id'])['probability'].cumsum()
        
        # Mark selected where rand < cum_prob and previous cum_prob <= rand
        df.loc[:,'prev_cum_prob'] = df.groupby(['person_id', 'selection_id'])['cum_prob'].shift(fill_value=0)
        df.loc[:,'selected'] = (df['rand'] >= df['prev_cum_prob']) & (df['rand'] < df['cum_prob'])
        
        # Clean up unnecessary columns
        df.drop(columns=['exp_util', 'probability', 'rand', 'cum_prob', 'prev_cum_prob'], inplace=True)

        return df
    
    @staticmethod
    def _multinomial_logit_selection_sobol(df):
        if Selector.considerMinimumUtility:
            df = df[df['utility'] > Selector.minimum_utility]
    
        df.loc[:, 'utility'] = np.minimum(df['utility'], Selector.maximum_utility)
    
        # Compute the probability
        df.loc[:, "exp_util"] = np.exp(df['utility'])        
        group_sum = df.groupby(['person_id', 'selection_id'])['exp_util'].transform('sum')
        df.loc[:, 'probability'] = df["exp_util"] / group_sum
    
        # Generate a Sobol number per group
        group_keys    = df.groupby(['person_id', 'selection_id']).ngroup()
        unique_groups = group_keys.unique()                        
        sobol_vals    = Selector.sobol_generator.random(len(unique_groups)).flatten()
        
        # Map Sobol values back to the groups
        sobol_map = pd.Series(sobol_vals, index=unique_groups)
        df.loc[:, 'group_key'] = group_keys
        df.loc[:, 'rand'] = df['group_key'].map(sobol_map)
    
        # Compute cumulative probability within each group
        df.loc[:, 'cum_prob'] = df.groupby(['person_id', 'selection_id'])['probability'].cumsum()
        df.loc[:, 'prev_cum_prob'] = df.groupby(['person_id', 'selection_id'])['cum_prob'].shift(fill_value=0)
    
        # Select where rand falls in the interval
        df.loc[:, 'selected'] = (df['rand'] >= df['prev_cum_prob']) & (df['rand'] < df['cum_prob'])
    
        # Clean up
        df.drop(columns=['exp_util', 'probability', 'rand', 'cum_prob', 'prev_cum_prob', 'group_key'], inplace=True)

        return df

    @staticmethod
    def _multinomial_logit_selection_not_efficient(df):        
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
