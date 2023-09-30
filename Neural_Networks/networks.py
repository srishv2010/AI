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
    def train(self, train_inputs, train_outputs, epochs=10, verbose=True):
        raise NotImplementedError("Network.train() has not been implemented yet.")
    
    def evaluate(self, test_inputs, test_outputs):
        raise NotImplementedError("Network.evaluate has not been implemented yet.")
    

###########################################
# Sequential Base Class ###################
###########################################
class Sequential(Network):
    def __init__(self, structure, loss_function=losses.MeanSquaredError) -> None:
        super().__init__(structure, loss_function)
        self.avg_loss = math.inf

    def predict(self, inputs):
        for layer in self.structure:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, train_inputs, train_outputs, epochs=10000, verbose=True, learning_rate=0.01,
              test_inputs=None, test_outputs=None):
        for epoch in range(1, epochs + 1, 1):
            total_loss = 0
            iterations = 0
            for inputs, outputs in zip(train_inputs, train_outputs):
                prediction = self.predict(inputs)
                iterations += 1
                total_loss += self.loss_function.loss(outputs, prediction)
                print(f"Error: {total_loss / iterations}; Epoch: {epoch};", end="\r")
                outputs_gradient = self.loss_function.loss_prime(outputs, prediction)
                for layer in reversed(self.structure):
                    outputs_gradient = layer.backward(outputs_gradient, learning_rate)
            test_error = self.evaluate(test_inputs, test_outputs, verbose=False)
            print(f"Train Data Error: {total_loss / iterations}; Test Data Error: {test_error}; Epoch: {epoch}/{epochs};")
    
    def evaluate(self, test_inputs, test_outputs, verbose=True):
        total_loss = 0
        iterations = 0
        for inputs, outputs in zip(test_inputs, test_outputs):
            iterations += 1
            prediction = self.predict(inputs)
            total_loss += self.loss_function.loss(outputs, prediction)
            if verbose:
                print(f"Error: {total_loss / iterations}", end="\r")
        self.avg_loss = total_loss / iterations
        if verbose:
            print(f"Error: {self.avg_loss}")
        return self.avg_loss
