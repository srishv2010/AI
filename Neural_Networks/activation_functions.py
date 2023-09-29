import numpy as np
import math as math
import time as time
from abc import abstractmethod


###########################################
# Activation Layer Base Class #############
###########################################
class ActivationLayer(object):
    def __init__(self) -> None:
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError("ActivationLayer.forward() has not been implemented yet.")
    
    @abstractmethod
    def backward(self, output_gradient, *args, **kwargs):
        raise NotImplementedError("ActivationLayer.backward() has not been implemented yet.")
    

###########################################
# Sigmoid Activation Layer Class ##########
###########################################
class Sigmoid(ActivationLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input):
        self.input = input
        self.output = 1/(1 + np.exp(-1 * self.input))
        return self.output

    def backward(self, output_gradient, *args, **kwargs):
        input_gradient = np.multiply(output_gradient, (self.output * (1 - self.output)))
        return input_gradient
