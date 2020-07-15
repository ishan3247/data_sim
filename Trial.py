import datetime
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plot
import math as m
from scipy.stats import poisson
from sklearn.preprocessing import normalize
from get_data import Data

class Trial:

    def __init__(self, trial_name, data):
        self.name = trial_name
        self._odor = int(self.find_odor())
        self._data = data
        self._total_spikes = None
        self._activation = False

    def odor(self):
        return self._odor

    def get_name(self):
        return self.name
    
    def find_odor(self):
        odor = self.name.split('O')
        return int(odor[1])

    def data(self):
        return self._data

    def sorption(self):
        if (self._odor == 7 or self._odor == 8 or self._odor == 9):
            return 0
        if (self._odor == 2 or self._odor == 1 or self._odor == 5):
            return 0
        if (self._odor == 6 or self._odor == 3 or self._odor == 4 or self._odor == 10):
            return 0

    def sorption_level(self):
        if (self._odor == 7 or self._odor == 8 or self._odor == 9):
            return "high"
        if (self._odor == 2 or self._odor == 1 or self._odor == 5):
            return "medium"
        if (self._odor == 6 or self._odor == 3 or self._odor == 4 or self._odor == 10):
            return "low"
    
    def set_total_spikes(self, sum):
        self._total_spikes = sum

    def get_total_spikes(self):
        return self._total_spikes

    def set_activation(self, bool):
        self._activation = bool

    def get_activation(self):
        return self._activation