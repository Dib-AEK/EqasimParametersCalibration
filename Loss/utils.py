# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 12:07:16 2025

@author: dabdelkader
"""
import polars as pl
import numpy as np

def get_shares(attr, modes):
    return {mode: attr.get(mode, 0.0) for mode in modes}



def compute_global_mode_share(selected_modes, modes):        
    counts_df = selected_modes.group_by("candidate_mode").agg(pl.count().alias("count"))
    total = counts_df["count"].sum()
    counts_df = counts_df.with_columns( (pl.col("count") / total).alias("share") )
    counts_dict = dict(zip(counts_df["candidate_mode"], counts_df["share"]))
    estimates_global_mode_share = {mode: [counts_dict.get(mode, 0.0),] for mode in modes}
    return estimates_global_mode_share


def compute_mode_share_distribution_by(selected_modes, modes, by = "canton_id"):
           
    pivoted = (selected_modes.select([by, "candidate_mode"])
                             .group_by([by, "candidate_mode"])
                             .agg(pl.len().alias("count"))
                             .pivot(values="count", index=by, on="candidate_mode", aggregate_function=None)
                             .sort(by)
                             .fill_null(0)                                 
                             .with_columns(
                                     pl.exclude(by)/ pl.sum_horizontal(pl.exclude(by))
                                     )
                             )                
    
    # Ensure all modes are present as columns
    missing_modes = [mode for mode in modes if mode not in pivoted.columns]
    if missing_modes:
        pivoted = pivoted.with_columns([pl.lit(0).alias(mode) for mode in missing_modes])
            
    return pivoted.select(modes).to_dict(as_series=False)


def compute_mode_distribution_by(selected_modes, modes, by="distance"):
    pivoted = (selected_modes.select([by, "candidate_mode"])
                             .group_by([by, "candidate_mode"])
                             .agg(pl.len().alias("count"))
                             .pivot(values="count", index=by, on="candidate_mode", aggregate_function=None)
                             .sort(by)
                             .fill_null(0)                                 
                             )                
    
    col_sums = pivoted.select(pl.exclude(by).sum())   
    pivoted = pivoted.with_columns(
                (pl.col(col) / col_sums[col][0]).fill_null(0) for col in pivoted.columns if col != by
            )
    
    # Ensure all modes are present as columns
    missing_modes = [mode for mode in modes if mode not in pivoted.columns]
    if missing_modes:
        pivoted = pivoted.with_columns([pl.lit(0).alias(mode) for mode in missing_modes])
            
    return pivoted.select(modes).to_dict(as_series=False)
    

    
    










