import datetime
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plot

np.random.seed(2)

def gen_train():
    dt = 0.001 # step
    frequency = 100 # 
    window = 0.1
    num_bins = window/dt

    array=[]

    for i in range(int(num_bins)):
        rand = random.random()
        if rand < frequency*dt: #frequency * dt is likelihood of firing
            #firing has ocurred
            array.append(1)
        else:
            array.append(0)

    return  array[0:50]

neuralData = np.random.random([8, 50])

colorCodes = np.array([[0, 0, 0],

                        [1, 0, 0],

                        [0, 1, 0],

                        [0, 0, 1],

                        [1, 1, 0],

                        [1, 0, 1],

                        [0, 1, 1],

                        [1, 0, 1]])

# Set spike colors for each neuron

lineSize = [0.4, 0.3, 0.2, 0.8, 0.5, 0.6, 0.7, 0.9]        

# Draw a spike raster plot

plot.eventplot(neuralData, color=colorCodes, linelengths = lineSize)   
# Provide the title for the spike raster plot

plot.title('Spike raster plot')
# Give x axis label for the spike raster plot
plot.xlabel('Neuron')
# Give y axis label for the spike raster plot
plot.ylabel('Spike')

# Display the spike raster plot

plot.show()

print(len(gen_train()))


train = np.array(gen_train())

print("train is: ", train)