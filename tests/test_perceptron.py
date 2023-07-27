"""
    Unit Test for Perceptron class in [./code/neural_network/perceptron]
"""
import numpy as np
import unittest
from wcode.neural_network.perceptron import Perceptron


class TestPerceptron(unittest.TestCase):
    """
    Perceptron class test
    """

    def test(self):
        """
        Tests the perceptrons ability to calculate to pass information
        """
        perceptron = Perceptron()
        self.assertEqual(type(perceptron), Perceptron, 'Simple Initialization has failed')
        self.assertEqual(perceptron.weigths, np.array([]),)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
