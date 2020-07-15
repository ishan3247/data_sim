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
from Trial import Trial
from Neuron import Neuron
from scipy.stats import iqr

set_size = 100
activation = 0.1
similarity = 0.8
nmb_odors = 10


# thresholds = [454.0, 455.0, 458.0, 456.0, 452.0, 456.0, 454.0, 454.0, 458.0, 456.0]

# 100 neurons, each neuron is an LNP neuron with 10 odors presented across 10 trials
neuron_set = []
for i in range(set_size):
    neuron = Neuron()
    neuron_set.append(neuron)

# run 1 (10 trials each odor once)
ensembles_1 = []
for odor in range(nmb_odors):
    ensembles_1.append(random.sample(range(0, set_size), int(activation*set_size)))
# print(ensembles_1)

# run 2 (10 trials each odor once), 80% neuron overlap, 20% different --> T to T variability 
ensembles_2 = []
for odor in range(nmb_odors):
    temp = []
    prev = []
    for i in range(int(nmb_odors*similarity)):
        rand = random.randint(0,9)
        if rand not in prev:
            temp.append(ensembles_1[odor][rand])
        prev.append(rand)
    prev = []
    ensembles_2.append(temp)
c = nmb_odors-int(nmb_odors*similarity)
for odor in range(nmb_odors):
    while c > 0:
        to_append = random.sample(range(0, set_size), c)
        for val in to_append:
            if val not in ensembles_2 and c != 0:
                c -= 1
                ensembles_2[odor].append(val)
# print(ensembles_2)

# RUN NMB 1
run_1 = []
odor = 1
for ensemble in ensembles_1:
    ens = []
    for neuron in ensemble:
        trials = neuron_set[neuron].get_trial_list()
        for trial in trials:
            if trial.odor() == odor:
                trial.set_activation(True)
        ens.append(neuron_set[neuron].run_trials())
    odor += 1
    run_1.append(ens)
print(len(run_1[0]))

# RUN NMB 1
run_2 = []
odor = 1
for ensemble in ensembles_2:
    ens = []
    for neuron in ensemble:
        trials = neuron_set[neuron].get_trial_list()
        for trial in trials:
            if trial.odor() == odor:
                trial.set_activation(True)
        ens.append(neuron_set[neuron].run_trials())
    odor += 1    
    run_2.append(ens)
print(len(run_2))



# set_size = 2000
# Make 100 neurons
# neuron_set = []
# for i in range(0, set_size):
#     neuron = Neuron()
#     temp = []
#     for trial in neuron.get_trial_list():
#         temp = neuron.run_trials()
#     neuron_set.append(temp)
# print(neuron_set)



# Find Q3, Q1, IQR
# stats = []
# for i in range(10):
#     temp = []
#     for neuron in neuron_set:
#         temp.append(neuron[i])
#     stats.append(np.percentile(temp, [25, 50, 75]))
# print(stats)

# threshold = []
# for i in range(len(stats)):
#     threshold.append(stats[i][2])
# print("hi")
# print(threshold)

# Find average spikes for each odor
# avg = []
# for i in range(10):
#     sum = 0
#     for neuron in neuron_set:
#         if neuron[i] > stats[i][0] - 1.5 * (stats[i][2]-stats[i][0]) and neuron[i] < stats[i][2] + 1.5 * (stats[i][2]-stats[i][0]):
#             sum += neuron[i]
#     avg.append(sum/set_size)
# print(avg)
