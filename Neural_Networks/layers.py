import numpy as np
import math as math
import time as time
from abc import abstractmethod


###########################################
# Layer Base Class ########################
###########################################
class Layer(object):
    def __init__(self, input_size, output_size) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.output = None
    
    @abstractmethod
    def forward(self, input):
        raise NotImplementedError("Layer.forward() has not been implemented yet.")
    
    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Layer.backward() has not been implemented yet.")


###########################################
# Dense Layer Class #######################
###########################################
class Dense(Layer):
    def __init__(self, input_size, output_size, weights=None, biases=None) -> None:
        super().__init__(input_size, output_size)
        self.weights = weights
        self.biases = biases
        
        if self.weights is None:
            self.weights = np.random.randn(output_size, input_size)

        if self.biases is None:
            self.biases = np.zeros(self.output_size)
    
    def forward(self, input):
        self.input = input
        weighted_input = np.dot(self.weights, self.input)
        self.output = weighted_input + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= weights_gradient * learning_rate
        self.biases -= output_gradient * learning_rate
        return input_gradient


