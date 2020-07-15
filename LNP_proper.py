import datetime
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plot
import math as m
from scipy.stats import poisson
from sklearn.preprocessing import normalize
from get_data import Trials

# get a time step (dt)
# get a freq
# get numb of bins

# take real data, find STA. Use STA to make linear filter

# for STA:  
# 1. Record the stimulus over complete spike train.
# 2. Record all spike times.
# 3. For every spike, add the stimulus values surrounding the spike into the spike-triggered average array. For example,
# a spike at 10.12s, the stimulus at 10.02s goes into the -0.10s bin, the stimulus at 10.03s goes into the -0.09s bin.
# Do this for a given time before and after the spike. Repeat for all spikes.
# 4. Once all the values are added into the array, divide by number of spikes to get spike-triggered average.

# take stim data. convolve w/lf

# take output + put thru non-lf

# take output use poisson spiking

T1 = Trials.getT1()
T2 = Trials.getT2()
T3 = Trials.getT3()
T1O8 = Trials.T1O8()
# print(T1O8)
T2O1 = Trials.T2O1()
T3O6 = Trials.T3O6()
T4O3 = Trials.T4O3()
T5O5 = Trials.T5O5()
T6O2 = Trials.T6O2()
T7O4 = Trials.T7O4()
T8O9 = Trials.T8O9()
T9O7 = Trials.T9O7()
T10O10 = Trials.T10O10()



# data = [T1[0:10],T2[0:10],T3[0:10]]
# data = [T1,T2,T3]
data = [T1O8, T2O1, T3O6, T4O3, T5O5, T6O2, T7O4, T8O9, T9O7, T10O10]
# print(data)


def linear_filter(data):
    filtered = []
    for trial in data:
        Ln = 0
        n = len(trial)
        for i in range(n):
            Ln += trial[n-1-i]*i*0.001
        filtered.append(Ln)
    return filtered

filtered = linear_filter(data)
print(filtered)

def norm_flow(filtered):
    norm = []
    for value in filtered:
        value = value/1000000
        norm.append(value)
    # counter = 0
    # biased = []
    # for value in norm:
    #     # if counter > 0:
    #         # value += 0.5
    #     biased.append(value)
    #     counter += 1
    return norm

norm = norm_flow(filtered)
print(norm)

def bias_flow(norm):
    norm[0] = norm[0]*0.2
    norm[1] = norm[1]*0.4
    norm[2] = norm[2]*0.4
    return norm

biased = bias_flow(norm)
print(biased)

def non_lf(biased):
    lam = []
    for nmb in biased:
        # out = float(1/(1+m.exp((-nmb))))
        out = float(np.log(1+m.exp(nmb)))
        # out = float(nmb**2)
        lam.append(out)
    return lam

lam = non_lf(biased)
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

def calc_nmb_spikes(spike_array):
    out = []
    for train in spike_array:
        sum = 0
        for nmb in train:
            if nmb > 0:
                sum += 1
        out.append(sum)
    return out

sum = calc_nmb_spikes(spikes)
print(sum)


tp1 = dist[0][100:200]
tp2 = dist[1][100:200]
tp3 = dist[2][100:200]
tp = [tp1,tp2,tp3]
# plot.eventplot(dist)  
# plot.show()