import unittest
import numpy as np
from wcode.activation.activation import Activation

class TestActivation(unittest.TestCase):

    def test(self):
        number = 0.6
        sigmoid_function = lambda x: 1 / (1 + np.e ** x)
        result = 0.35434369377420455

        activation = Activation()
        activation_with_function = Activation(sigmoid_function)

        self.assertEqual(activation.compute(number), number, "Default activation function is failing")
        self.assertAlmostEqual(activation_with_function.compute(number), result, 16, "Sigmoid activation function is not working")