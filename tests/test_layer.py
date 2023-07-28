import unittest
import numpy as np
from wcode.layers.layer import Layer
from numpy.testing import assert_array_equal

class TestLayer(unittest.TestCase):

    def test(self):
        array = [1, 2]
        result = [5/4, 2]
        test_function = lambda x: (x/2)**2 + 1

        layer = Layer()
        layer_with_custom_function = Layer(test_function)

        assert_array_equal(layer.predict(array), array, "Default Layer function is failing.")
        self.assertIsInstance(layer.predict(array), np.ndarray, "List was not converted to Numpy Array")
        assert_array_equal(layer_with_custom_function.predict(array), result, "Layer with custom function is failing.")