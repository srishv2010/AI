import numpy as np
import scipy.signal as sgl
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
    def __init__(self, inputs_shape) -> None:
        self.inputs = None
        self.outputs = None
        self.inputs_shape = inputs_shape
        
    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError("Convolutional.forward() has not been implemented yet.")
    
    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Convolutional.backward() has not been implemented yet.")


###################################
# Convolutional2D Layer Class #####
###################################
class Convolutional2D(Convolutional):
    def __init__(self,
                 inputs_shape,
                 kernel_shape,
                 depth=1,
                 kernels=None,
                 biases=None
                 ) -> None:
        super().__init__(inputs_shape)
        self.inputs_depth, self.inputs_height, self.inputs_width = self.inputs_shape
        self.kernels_shape = (
            depth,
            self.inputs_depth,
            kernel_shape[0],
            kernel_shape[1]
        )
        self.outputs_shape = (
            depth,
            self.inputs_height - kernel_shape[0] + 1,
            self.inputs_width - kernel_shape[1] + 1
        )
        self.kernels = kernels
        self.biases = biases
        if self.kernels is None:
            self.kernels = np.random.randn(*self.kernels_shape)
        if self.biases is None:
            self.biases = np.random.randn(*self.outputs_shape)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.copy(self.biases)
        for i, kernels in enumerate(self.kernels):
            for j, kernel in enumerate(kernels):
                self.outputs[i] += sgl.correlate2d(self.inputs[j], kernel, "valid")
        return self.outputs
    
    def backward(self, outputs_gradient, learning_rate):
        inputs_gradient = np.zeros(self.inputs_shape)
        kernels_gradient = np.zeros(self.kernels_shape)
        
        for i, output_gradient in enumerate(outputs_gradient):
            for j, inputs in enumerate(self.inputs):
                kernels_gradient[i, j] = sgl.correlate2d(
                    inputs,
                    output_gradient,
                    "valid"
                )
                inputs_gradient[j] += sgl.convolve2d(
                    output_gradient,
                    self.kernels[i, j],
                    "full"
                )
        
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * outputs_gradient
        return inputs_gradient


###################################
# Reshape Layer Class #############
###################################
class Reshape(object):
    def __init__(self, output_shape) -> None:
        self.output_shape = output_shape
        self.inputs_shape = None
    
    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(self.output_shape)
    
    def backward(self, output_gradient, *args, **kwargs):
        return np.reshape(output_gradient, self.inputs_shape)
