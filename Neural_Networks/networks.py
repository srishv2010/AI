import numpy as np
import math as math
import time as time
from abc import abstractmethod
import layers as layers
import losses as losses
import activation_functions as actv_func




###########################################
# Network Base Class ######################
###########################################
class Network(object):
    def __init__(self, layers, loss_function=losses.MeanSquaredError) -> None:
        self.layers = layers
        self.loss_function = loss_function
    
    @abstractmethod
    def predict(self, input):
        raise NotImplementedError("Network.predict() has not been implemented yet.")
    
    @abstractmethod
    def train(self, input_train, output_train, epochs=10, verbose=True):
        raise NotImplementedError("Network.train() has not been implemented yet.")
    
    '''
    def evaluate(self, input_test, output_test):
        raise NotImplementedError("Network.evaluate has not been implemented yet.")
    '''


###########################################
# Sequential Base Class ###################
###########################################
class Sequential(Network):
    def __init__(self, layers, loss_function=losses.MeanSquaredError) -> None:
        super().__init__(layers, loss_function)

    def predict(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def train(self, input_train, output_train, epochs=10000, verbose=True, learning_rate=0.01):
        for epoch in range(1, epochs + 1, 1):
            total_loss = 0
            iterations = 0
            for input, output in zip(input_train, output_train):
                prediction = self.predict(input)
                iterations += 1
                total_loss += self.loss_function.loss(output, prediction)

                output_gradient = self.loss_function.loss_prime(output, prediction)
                for layer in reversed(self.layers):
                    output_gradient = layer.backward(output_gradient, learning_rate)
            print(f"Accuracy: {100 * (1 - (total_loss / iterations))}%; Epoch: {epoch}")
            