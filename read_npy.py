# !pip install numpy==1.16.1
import numpy as np
import matplotlib.pyplot as plt

data = np.load('Fneu.npy')
plt.plot(data[0])
plt.show()


print(data)