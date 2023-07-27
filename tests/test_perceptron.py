"""
    Unit Test for Perceptron class in [./code/neural_network/perceptron]
"""
import unittest
from wcode.neural_network.perceptron import Perceptron

class PerceptronTest(unittest.TestCase):
    """
        Perceptron class test
    """

    def test(self):
        """ 
            Tests the perceptrons ability to calculate to pass information
        """
        perceptron = Perceptron()
        self.assertEqual(type(perceptron), Perceptron, 'Simple Initialization has failed')


if __name__ == '__main__':
    unittest.main(warnings='ignore')
