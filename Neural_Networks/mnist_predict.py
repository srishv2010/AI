import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import pickle
import matplotlib.pyplot as plt


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt


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
x_test, y_test = preprocess_data(x_test, y_test, 10)

with open('mnist_model.pkl', 'rb') as inp:
    network = pickle.load(inp)
    for x, y in zip(x_test, y_test):
        img = gen_image(x)
        img.title(f"Prediction: {np.argmax(network.predict(x))}, True: {np.argmax(y)}")
        img.show()
