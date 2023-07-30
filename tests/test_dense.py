from wcode.exceptions.exceptions import NoArgumentChosenException
from wcode.neural_network.perceptron import Perceptron
from wcode.layers.dense import Dense
import numpy as np
import unittest


class TestDense(unittest.TestCase):

    def test(self):
        perceptrons = [Perceptron(), Perceptron(), Perceptron()]
        number_of_neurons = 3
        arguments = [1,2,3]
        result = sum(arguments)
        bias = 1.5

        with self.assertRaises(NoArgumentChosenException):
            dense = Dense()

        dense = Dense(number_of_neurons)
        self.assertEqual(dense.shape, number_of_neurons, "Number of Perceptrons by providing argument number_of_neurons is not the expected value")

        dense1 = Dense(neurons=perceptrons)
        self.assertEqual(dense1.shape, number_of_neurons, "Number of Perceptrons by providing argument neurons is not the expected value")

        dense2 = Dense(number_of_neurons)
        self.assertEqual(dense2.compute(arguments), result, "")
        

