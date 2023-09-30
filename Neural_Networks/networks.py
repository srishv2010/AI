import numpy as np
import math as math
import time as time
from abc import abstractmethod
import layers as layers
import losses as losses
import activation_functions as activation_functions


###########################################
# Network Base Class ######################
###########################################
class Network(object):
    def __init__(self, structure, loss_function=losses.MeanSquaredError) -> None:
        self.structure = structure
        self.loss_function = loss_function
    
    @abstractmethod
    def predict(self, inputs):
        raise NotImplementedError("Network.predict() has not been implemented yet.")
    
    @abstractmethod
    def train(self, train_inputs, output_train, epochs=10, verbose=True):
        raise NotImplementedError("Network.train() has not been implemented yet.")
    
    '''
    def evaluate(self, inputs_test, output_test):
        raise NotImplementedError("Network.evaluate has not been implemented yet.")
    '''


###########################################
# Sequential Base Class ###################
###########################################
class Sequential(Network):
    def __init__(self, structure, loss_function=losses.MeanSquaredError) -> None:
        super().__init__(structure, loss_function)

    def predict(self, inputs):
        for layer in self.structure:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, train_inputs, output_train, epochs=10000, verbose=True, learning_rate=0.01):
        for epoch in range(1, epochs + 1, 1):
            total_loss = 0
            iterations = 0
            for inputs, output in zip(train_inputs, output_train):
                prediction = self.predict(inputs)
                iterations += 1
                total_loss += self.loss_function.loss(output, prediction)

                output_gradient = self.loss_function.loss_prime(output, prediction)
                for layer in reversed(self.structure):
                    output_gradient = layer.backward(output_gradient, learning_rate)
            print(f"Error: {total_loss / iterations}; Epoch: {epoch};")
            