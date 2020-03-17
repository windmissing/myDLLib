import torch
import numpy as np

class LinearUnit(object):

    @staticmethod
    def activation(x):
        return x

    @staticmethod
    def derivative(x):
        return 1

    def cross_entropy(self, target, predict):
        return 0.5*np.linalg.norm(target-predict)**2

    def derivative_cross_entropy(self, target, predict):
        return -target / predict + np.nan_to_num((1 - target) / (1 - predict))

class SigmoidUnit(object):
    
    @staticmethod
    def activation(x):
        return 1/(1+torch.exp(-x))
    
    @staticmethod
    def derivative(x):
        return x*(1-x)

    def cross_entropy(self, target, predict):
        return np.sum(np.nan_to_num(-target * np.log(predict) - (1 - target) * np.log(1 - predict)))

    def derivative_cross_entropy(self, target, predict):
        return -target / predict + np.nan_to_num((1 - target) / (1 - predict))