"""
    Unit Test for Perceptron class in [./code/neural_network/perceptron]
"""
import numpy as np
import unittest
from wcode.neural_network.perceptron import Perceptron
from wcode.neural_network.perceptron import ArrayMismatchException
from numpy.testing import assert_equal


class TestPerceptron(unittest.TestCase):
    """
    Perceptron class test
    """

    def test(self):
        """
        Tests the perceptrons ability to calculate to pass information
        """
        empty_array = np.array([])
        wrong_array = [1]
        weigths = [1, -2, 0, 4]
        X = [4, 5, -1, 0]
        bias=3
        second_bias = 0

        perceptron = Perceptron(bias=bias)
        second_perceptron = Perceptron(weigths, second_bias)

        self.assertEqual(type(perceptron), Perceptron, 'Simple Initialization has failed.')
        self.assertEqual(type(perceptron.weigths), type(np.array([])), "List was not converted to Numpy array.")
        assert_equal(perceptron.weigths, empty_array, "Numpy Arrays are not equal.")

        with self.assertRaises(ArrayMismatchException, msg="Array Match is not being checked."):
            perceptron.predict(wrong_array)

        self.assertEqual(perceptron.predict([]), 3, "Compute is not taking bias into consideration.")

        self.assertEqual(second_perceptron.predict(X), -6, "Compute matrix multiplication is failing.")

        

if __name__ == "__main__":
    unittest.main()
