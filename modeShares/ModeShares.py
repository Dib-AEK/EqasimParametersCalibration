#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 09:59:30 2025

@author: dabdelkader
"""

from abc import ABC, abstractmethod

class ModeShares(ABC):
    @abstractmethod
    def get_distance_bins(self):
        pass

    @abstractmethod
    def get_distance_labels(self):
        pass

    @abstractmethod
    def get_mode_shares(self):
        pass

    @abstractmethod
    def get_mode_share_by(self, by):
        pass