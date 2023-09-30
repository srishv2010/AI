from layers import Dense
import networks as networks
import losses as losses
from activation_functions import Sigmoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as time
import math as math


inputs = np.resize([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], (4, 2, 1))

output = np.reshape([
    [0], 
    [1], 
    [1], 
    [0]
], (4, 1, 1))


network = networks.Sequential([
    Dense(2, 3),
    Sigmoid(),
    Dense(3, 1)
], loss_function=losses.MeanSquaredError)

points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = network.predict([[x], [y]])
        points.append([x, y, z[0, 0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")


network.train(inputs, output)


points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = network.predict([[x], [y]])
        points.append([x, y, z[0, 0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()
