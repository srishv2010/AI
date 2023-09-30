import numpy as np
import math as math
import time as time
from abc import abstractmethod


###########################################
# Layer Base Class ########################
###########################################
class Layer(object):
    def __init__(self, inputs_size, output_size) -> None:
        self.inputs_size = inputs_size
        self.output_size = output_size
        self.inputs = None
        self.output = None
    
    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError("Layer.forward() has not been implemented yet.")
    
    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Layer.backward() has not been implemented yet.")


###########################################
# Dense Layer Class #######################
###########################################
class Dense(Layer):
    def __init__(self, inputs_size, output_size, weights=None, biases=None) -> None:
        super().__init__(inputs_size, output_size)
        self.weights = weights
        self.biases = biases
        
        if self.weights is None:
            self.weights = np.random.randn(output_size, inputs_size)

        if self.biases is None:
            self.biases = np.zeros((self.output_size, 1))
    
    def forward(self, inputs):
        self.inputs = inputs
        weighted_inputs = np.dot(self.weights, self.inputs)
        self.output = weighted_inputs + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        inputs_gradient = np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.inputs.T)
        self.weights -= weights_gradient * learning_rate
        self.biases -= output_gradient * learning_rate
        return inputs_gradient


###################################
# Convolutional Layer Base Class ##
###################################
class Convolutional(object):
    def __init__(self, inputs_shape, kernel_shape, depth) -> None:
        self.inputs_shape = inputs_shape
        self.inputs_depth, self.inputs_height, self.inputs_width = self.inputs_shape
        self.kernels_shape = (depth, self.inputs_depth, kernel_shape[0], kernel_shape[1])
        self.outputs_shape = (depth, self.inputs_height - kernel_shape[0] + 1, self.inputs_width - kernel_shape[1] + 1)
        