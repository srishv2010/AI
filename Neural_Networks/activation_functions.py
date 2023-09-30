import numpy as np
import math as math
import time as time
from abc import abstractmethod


###########################################
# Activation Layer Base Class #############
###########################################
class ActivationLayer(object):
    def __init__(self) -> None:
        self.inputs = None
        self.output = None

    @abstractmethod
    def forward(self, inputs):
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
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-1 * self.inputs))
        return self.output

    def backward(self, output_gradient, *args, **kwargs):
        inputs_gradient = np.multiply(output_gradient, (self.output * (1 - self.output)))
        return inputs_gradient
