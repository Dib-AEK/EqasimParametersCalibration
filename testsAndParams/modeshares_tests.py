# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:03:30 2025

@author: dabdelkader
"""
import os
os.chdir("..")

from modeShares.modeShares import ModeShares
import matplotlib.pyplot as plt


eqasim_cache_path = "Z:\ch-zh-synpop/cache10p100"
ms = ModeShares(eqasim_cache_path, overwrite=False)



#%%
for k,v in ms.mode_shares["mode_distance"].items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()    