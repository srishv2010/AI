from layers import Dense
import networks as networks
import losses as losses
from activation_functions import Sigmoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time
import math as math
from keras.datasets import mnist
from keras.utils import np_utils
import pickle


def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 60000)
x_test, y_test = preprocess_data(x_test, y_test, 10000)

network = networks.Sequential([
    Dense(28 * 28, 64),
    Sigmoid(),
    Dense(64, 16),
    Sigmoid(),
    Dense(16, 16),
    Sigmoid(),
    Dense(16, 10),
    Sigmoid()
], loss_function=losses.MeanSquaredError)

network.train(x_train, y_train,  500, True, 0.01, x_test, y_test,)
for x, y in zip(x_test, y_test):
    output = network.predict(x)
    print('Prediction:', np.argmax(output), 'Actual:', np.argmax(y))

network.evaluate(x_test, y_test)
