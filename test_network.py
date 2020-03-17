import unittest

from src.mnist_loader import load_data_shared
from src.cost import QuadraticCost
from src.cost import CrossEntropy
from src.unit import SigmoidUnit, LinearUnit
from src.layer import Layer
from src.network import Network

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class TestMyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_data, cls.validation_data, cls.test_data = load_data_shared(filename="../mnist.pkl.gz",
                                                                     seed=666,
                                                                     train_size=2000,
                                                                     vali_size=0,
                                                                     test_size=100)
    def test_6_2_1_1(self):
        nn1 = Network(cost=QuadraticCost(), utmode=True)
        nn1.fit(self.training_data[0], self.training_data[1], self.test_data[0],
                                                            self.test_data[1])
        nn2 = Network(cost=CrossEntropy(), utmode=True)
        nn2.fit(self.training_data[0], self.training_data[1], self.test_data[0],
                                                            self.test_data[1])
    def test_6_2_2_1(self):
        nn = Network(layer=[Layer(LinearUnit(), 784), Layer(LinearUnit(), 30), Layer(LinearUnit(), 10)],
                     cost=CrossEntropy(),
                     utmode=True)
        nn.fit(self.training_data[0], self.training_data[1],
               self.test_data[0], self.test_data[1],
               learning_rate=7 * 1e-7)


if __name__ == '__main__':
    unittest.main()
