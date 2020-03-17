import torch
from sklearn.metrics import accuracy_score
import numpy as np
from src.cost import QuadraticCost
from src.cost import CrossEntropy
from src.unit import SigmoidUnit, LinearUnit
from src.layer import Layer

class Network(object):
    
    def __init__(self,
                 layer=[Layer(LinearUnit(), 784), Layer(SigmoidUnit(), 30), Layer(SigmoidUnit(), 10)],
                 cost=QuadraticCost(),
                 log=False, utmode=False):
        # Variables initialization
        self.inputlayer_neurons = layer[0].num
        self.hiddenlayer = layer[1]
        self.outputlayer = layer[2]
        self.cost = cost
        self.cost.setOutputUnit(self.outputlayer.unit)
        self.log = log
        self.utmode = utmode

        # weight and bias initialization
        self.hiddenlayer.init_parameters(self.inputlayer_neurons)
        self.outputlayer.init_parameters(self.hiddenlayer.num)

    def to_number(self, vec):
        return [np.argmax(v) for v in vec]

    def fit(self, X, y, testX, testY, epoch = 1000, learning_rate=1e-3):
        # Variables initialization
        if self.utmode: epoch = 1
        else: epoch = epoch
        #learning_rate = 1e-3
        return self.GD(X, y, testX, testY, epoch, learning_rate)
    
    def GD(self, X, y, testX, testY, epoch, learning_rate):
        test_scores = []
        train_scores = []
        cost_scores = []
        for i in range(epoch):
            # Forward Propogation            
            self.forward(X)
            cost_scores.append(self.cost.fn(torch.Tensor(y), self.outputlayer.a))
            
            # Backword Propogation
            self.backward(X, y, learning_rate)
            
            if self.log:
                print ("the ", i, "times")
            test_scores.append(self.predict(testX, testY))
            train_scores.append(self.predict(X, y))
        return (cost_scores, np.array(train_scores), np.array(test_scores))
            
    def forward(self, X, isSave = True):
        hidden_a = self.hiddenlayer.calc_z_a(X, isSave)
        output_a = self.outputlayer.calc_z_a(hidden_a, isSave)
        return output_a
        
    def backward(self, X, y, learning_rate):
        self.outputlayer.gh = - self.cost.derivative( torch.Tensor(y), self.outputlayer.a)
        self.outputlayer.calc_gz()
        self.hiddenlayer.gh = self.outputlayer.calc_gh_for_pre_layer()
        self.hiddenlayer.calc_gz()
        self.outputlayer.update_parameters(self.hiddenlayer.a,learning_rate)
        self.hiddenlayer.update_parameters(X, learning_rate)

    def predict(self, testX, testY):
        out_a_test = self.forward(testX, False)
        accuracy = accuracy_score(self.to_number(testY), self.to_number(out_a_test))
        if self.log:
            print (accuracy)
        return accuracy