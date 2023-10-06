import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from layers import Dense, Reshape, Convolutional2D
from activation_functions import Sigmoid
from losses import BinaryCrossEntropy
from networks import Sequential


def preprocess_data(x, y, limit):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 10000)

network = Sequential([
    Convolutional2D((1, 28, 28), (5, 5), 5),
    Sigmoid(),
    Convolutional2D((5, 24, 24), (3, 3), 5),
    Sigmoid(),
    Reshape((22 * 22 * 5, 1)),
    Dense(22 * 22 * 5, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
], loss_function=BinaryCrossEntropy)

# train
network.train(
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = network.predict(x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
