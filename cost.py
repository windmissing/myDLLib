import numpy as np

class CostFunction(object):
    def setOutputUnit(self, unit):
        self.outputUnit = unit

class QuadraticCost(CostFunction):

    def fn(self, target, predict):
        return 0.5*np.linalg.norm(target-predict)**2

    def derivative(self, target, predict):
        return predict - target

class CrossEntropy(CostFunction):

    def fn(self, target, predict):
        return self.outputUnit.cross_entropy(target, predict)

    def derivative(self, target, predict):
        return self.outputUnit.derivative_cross_entropy(target, predict)