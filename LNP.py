import datetime
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plot
import math as m
from scipy.stats import poisson
from sklearn.preprocessing import normalize

# np.random.seed(2)
dt = 0.001 # step
frequency = 4 # 
window = 1
data = np.load('Fneu.npy')
num_bins = len(data[0])

def gen_train(P):
    array=[]
    for p in P:
        rand = random.random()
        if rand < p: #frequency * dt is likelihood of firing
            #firing has ocurred
            array.append(1)
        else:
            array.append(0)
    return  array


# takes in an array of 1s and 0s and returns an array with the times of each spike
def calculate_distance(spike_array):
    timestamp =0.0
    spike_time = []

    for i in range(len(spike_array)):
        if spike_array[i] >= 1:
            spike_time.append(timestamp)
        timestamp += dt

    return spike_time

def tk():
    array=[]
    for i in range(int(num_bins)):
        array.append(i)
    # print(array)
    return array

tk = tk()

def stimData():
    array=[]
    for i in range(int(num_bins)):
        rand = random.random()
        array.append(rand*i)
    # print(array)
    return array

# stimData = stimData()
stimData = data[0]
# print(stimData)

def linFilter():
    array=[]
    # counter = 0
    for i in tk:
        # filt1 = float(m.exp(-((nmb+num_bins/5)/(num_bins/12))**2)-.25*m.exp(-((nmb+num_bins/2)/(num_bins/5))**2))
        # y = m.cos(i/50)*320
        y = (i**3)/1000000
        array.append(y)
        # counter += 0.5
    # print(array)
    return array

linear_filter = linFilter()
# array = []
# for v in linear_filter:
#     normalized_v = v / np.sqrt(np.sum(v**2))
#     array.append(normalized_v)
# print (array)

# toPlot=[]
toPlot = [stimData, linear_filter]
# plot.plot(toPlot)
# plot.show()
# plot.plot(linear_filter)
# plot.show()

def convolve(stimData, linear_filter):
    # print(stimData)
    # if (stimData != None) and (linear_filter != None):
    return np.convolve(stimData, linear_filter, mode='full')

# print("hi")
convolved = convolve(stimData, linear_filter)
# print(convolved)
# plot.plot(convolved)
# plot.show()

def nonLinFilter(convolved):
    array=[]
    for nmb in range(num_bins):
        # try:
        #     out = float(1/(1+m.exp((-nmb))))
        # except OverflowError:
        #     out = float('inf')
        # out = 10*(nmb)**2
        y = float(-8000*((nmb-13000)**2)+970000000000)
        if y > 0:
            array.append(y)
        else:
            array.append(0)
    return array

nl = nonLinFilter(convolved)
# print(nl)
# plot.plot(nl)
# plot.show()

def poisson(nl):
    # array=[]
    # for nmb in nl:
    #     lam = ((nmb**frequency) * m.exp(-nmb))/m.factorial(frequency)
    #     array.append(lam)
    array = np.random.poisson(nl, size=None)
    return array

P = poisson(nl)
print(P[0:10000])

# train = gen_train(P)
# dist = calculate_distance(P)
# print(dist[0:100])
plot.eventplot(P[0:10000])  
plot.show()