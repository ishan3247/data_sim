import datetime
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plot

np.random.seed(2)
dt = 0.001 # step
frequency = 4 # 
window = 1
num_bins = window/dt

def gen_train(frequency):
    array=[]
    for i in range(int(num_bins)):
        rand = random.random()
        if rand < frequency*dt: #frequency * dt is likelihood of firing
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
        if spike_array[i] == 1:
            spike_time.append(timestamp)
        timestamp += dt

    return spike_time


# frequency with slow breathing
self_data_slow = calculate_distance(gen_train(4))
# frequency with fast breathing
self_data_fast = calculate_distance(gen_train(12))
# plot data
self_data = [self_data_slow, self_data_fast]
plot.eventplot(self_data)  
plot.show()
# # plot.eventplot(self_data)  




# # print(self_data)

# #print()

# # neuralData = np.random.random([8, 50])

# # colorCodes = np.array([[0, 0, 0],

# #                         [1, 0, 0],

# #                         [0, 1, 0],

# #                         [0, 0, 1],

# #                         [1, 1, 0],

# #                         [1, 0, 1],

# #                         [0, 1, 1],

# #                         [1, 0, 1]])

# # # Set spike colors for each neuron
# # lineSize = [0.2, 0.3, 0.2, 0.8, 0.5, 0.6, 0.7, 0.9]        

# # # Draw a spike raster plot

# plot.eventplot(self_data)   
# # # Provide the title for the spike raster plot

# # plot.title('Spike raster plot')
# # # Give x axis label for the spike raster plot
# # plot.xlabel('Neuron')
# # # Give y axis label for the spike raster plot
# # plot.ylabel('Spike')

# # # Display the spike raster plot

# plot.show()

# # print(len(gen_train()))


# # train = np.array(gen_train())

# # print("train is: ", train)