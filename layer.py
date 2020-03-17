import torch
import numpy as np

class Layer(object):
    def __init__(self, unit, num):
        self.unit = unit
        self.num = num
        self.unit.layer = self
        
    def init_parameters(self, input_num):
        self.weights = torch.randn(input_num, self.num).type(torch.FloatTensor)
        self.bias = torch.randn(1, self.num).type(torch.FloatTensor)
        
    def calc_z_a(self, X, isSave = True):
        z = torch.mm(torch.Tensor(X), self.weights) + self.bias
        a = self.unit.activation(z)
        if isSave:
            self.z = z
            self.a = a
        return a
    
    def calc_gz(self):
        #print (self.gh.shape, self.unit.derivative(self.a).shape)
        self.gz = self.gh * self.unit.derivative(self.a)
    
    def calc_gh_for_pre_layer(self):
        # 书上是这样的
        # return torch.mm(self.weights.t(), self.gz)
        # 网上的例子是这样的
        # [TODO:为什么不一样呢]
        #print (self.gz.data.shape, self.weights.shape)
        return torch.mm(torch.Tensor(self.gz), self.weights.t())
    
    def update_parameters(self, X, learning_rate):
        self.bias += self.gz.sum() * learning_rate
        self.weights += torch.mm(torch.Tensor(X).t(), self.gz) * learning_rate

    @staticmethod
    def cross_entropy(target):
        return np.sum(np.nan_to_num(-target * np.log(predict) - (1 - target) * np.log(1 - predict)))

    @staticmethod
    def derivative_cross_entropy(target, predict):
        return -target / predict + np.nan_to_num((1 - target) / (1 - predict))