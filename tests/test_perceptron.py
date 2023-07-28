"""
    Unit Test for Perceptron class in [./code/neural_network/perceptron]
"""
import numpy as np
import unittest
from wcode.neural_network.perceptron import Perceptron
from numpy.testing import assert_equal


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
        self.assertEqual(type(perceptron.weigths), type(np.array([])), "List was not converted to Numpy array")

        assert_equal(perceptron.weigths, np.array([1]), "Numpy Arrays are not equal")


if __name__ == "__main__":
    unittest.main(warnings="ignore")
