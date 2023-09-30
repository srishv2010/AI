import numpy as np
import math as math
import time as time
from abc import abstractmethod


###########################################
# Cost Function Base Class ################
###########################################
class LossFunction(object):
    @staticmethod
    @abstractmethod
    def loss(output_true, output_prediction):
        raise NotImplementedError("LossFunction.loss() has not been implemented yet")
    
    @staticmethod
    @abstractmethod
    def loss_prime(output_true, output_prediction):
        raise NotImplementedError("LossFunction.loss_prime() has not been implemented yet")
    
    
###########################################
# Mean Squared Error Cost Function Class ##
###########################################
class MeanSquaredError(LossFunction):
    @staticmethod
    def loss(output_true, output_prediction):
        loss = np.mean(np.power((output_true - output_prediction), 2))
        return loss

    @staticmethod 
    def loss_prime(output_true, output_prediction):
        loss_prime = 2 * (output_prediction - output_true) / np.size(output_prediction)
        return loss_prime
