import numpy as np
import pandas as pd
import math as math
import time as time
from abc import abstractmethod


###################################
# Layer Base Class ################
###################################
class Layer(object):
    def __init__(self, input_size, output_size) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.inputs = None
        self.outputs = None
    
    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError("Layer.forward() has not been implemented yet.")
    
    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Layer.backward() has not been implemented yet.")


###################################
# Dense Layer Class ###############
###################################
class Dense(Layer):
    def __init__(self, input_size, output_size, weights=None, biases=None):
        super().__init__(input_size, output_size)
        self.weights = weights
        self.biases = biases
        
        if self.weights is None:
            self.weights = np.random.randn(output_size, input_size)

        if self.biases is None:
            self.biases = np.zeros(self.output_size)
    
    def forward(self, inputs):
        self.inputs = inputs
        weighted_inputs = np.dot(self.weights, self.inputs)
        self.outputs = weighted_inputs + self.biases
        return self.outputs
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot()
        
