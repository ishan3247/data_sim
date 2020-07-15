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

nmb_odors = 10
nmb_data_per_trial = 1000
nmb_trials = 10

# 1s-2s (can go higher) # odor c.50ms after presentation goes to brain
# get nwb from max, get data with timestamps, can start with cleaner data
# dont bias to odor
# Representations of odor in the piriform cortex. Stettler DD, Axel R. Neuron. 2009
# read the papers, v important

class Trial:

    def __init__(self, trial_name, data):
        self.name = trial_name
        self._odor = int(self.find_odor())
        self._data = data
        self._total_spikes = None

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

T1O8 = Trial("T1O8", Data.T1O8())
T2O1 = Trial("T2O1", Data.T2O1())
T3O6 = Trial("T3O6", Data.T3O6())
T4O3 = Trial("T4O3", Data.T4O3())
T5O5 = Trial("T5O5", Data.T5O5())
T6O2 = Trial("T6O2", Data.T6O2())
T7O4 = Trial("T7O4", Data.T7O4())
T8O9 = Trial("T8O9", Data.T8O9())
T9O7 = Trial("T9O7", Data.T9O7())
T10O10 = Trial("T10O10", Data.T10O10())


data = [T1O8.data(), T2O1.data(), T3O6.data(), T4O3.data(), T5O5.data(), T6O2.data(), T7O4.data(), T8O9.data(), T9O7.data(), T10O10.data()]
trial_list = [T1O8, T2O1, T3O6, T4O3, T5O5, T6O2, T7O4, T8O9, T9O7, T10O10]

def trial_matrix(trial_list):
    array = [[] for i in range(nmb_trials)]
    c = 0
    for trial in trial_list:
        for i in range(nmb_trials):
            if trial.odor() == i+1:
                array[c].append(1)
            else:
                array[c].append(0)
        c += 1
    return array

trial_matrix = trial_matrix(trial_list)
flow_matrix = data

def concatenate_horizontal(m1, m2):
    return np.concatenate((m1, m2), axis=1)

to_lf = concatenate_horizontal(trial_matrix, flow_matrix)

def temp_kernel_matrix():
    array = []
    for i in range(nmb_data_per_trial):
            # array.append((nmb_data_per_trial-1-i)*0.001)
            array.append(1)
    return array

temp_kernel_matrix = temp_kernel_matrix()

def bias_matrix(trial_list):
    array = [0]*nmb_odors
    for trial in trial_list:
        array[trial.odor()-1] = trial.sorption()
    return array

bias_matrix = bias_matrix(trial_list)

def coeff_matrix(bias_matrix, temp_kernel_matrix):
    array = []
    for coeff in bias_matrix:
        array.append(coeff)
    for coeff in temp_kernel_matrix:
        array.append(coeff)  
    return array

coeff_matrix = coeff_matrix(bias_matrix, temp_kernel_matrix)

def L_stage(to_lf, coeff_matrix):
    # print(to_lf[0])
    # print(coeff_matrix)
    return np.matmul(to_lf,coeff_matrix)

lf_matrix = L_stage(to_lf, coeff_matrix)
print("here")
print(lf_matrix)

def norm_L_stage(lf_matrix):
    norm = []
    for nmb in lf_matrix:
        norm.append(nmb/1000000)
    return norm

norm_L_stage = norm_L_stage(lf_matrix)
print(norm_L_stage)

def N_stage(norm_L_stage):
    lam = []
    for nmb in norm_L_stage:
        out = float(1/(1+m.exp((-nmb))))
        # out = float(np.log(1+m.exp(nmb)))
        # out = float(nmb**2)
        lam.append(out)
    return lam

lam = N_stage(norm_L_stage)
print(lam)

def gen_spikes(lam):
    spike = []
    for l in lam:
        spike.append(np.random.poisson(l, size=len(data[0])))
    return spike

spikes = gen_spikes(lam)
# print(spikes)

def calculate_distance(spike_array):
    out = []
    for train in spike_array:
        spike_time = []
        timestamp = float(0.0)
        for i in range(len(train)):
            if train[i] > 0:
                for i in range(1, train[i]+1):
                    spike_time.append(timestamp)
            timestamp += float(0.001)
        out.append(spike_time)
    return out

dist = calculate_distance(spikes)
# print(dist)

def calc_nmb_spikes(spike_array, trial_list):
    # out = []
    c = 0
    for train in spike_array:
        sum = 0
        for nmb in train:
            if nmb > 0:
                sum += 1
        trial_list[c].set_total_spikes(sum)
        c += 1
        # out.append(sum)
    # return out

calc_nmb_spikes(spikes, trial_list)

def to_print(trial_list):
    for trial in trial_list:
        print("trial: " + str(trial.get_name()) + "; odor: " + str(trial.odor()) + "; sorption: " + trial.sorption_level() + "; spikes: " + str(trial.get_total_spikes()))


to_print = to_print(trial_list)
# print(to_print)


# tp1 = dist[0][100:200]
# tp2 = dist[1][100:200]
# tp3 = dist[2][100:200]
# tp = [tp1,tp2,tp3]
plot.eventplot(dist)  
plot.show()