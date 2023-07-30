"""
    This module contains everything needed to create a single perceptron
"""
from typing import List
from numbers import Number
from dataclasses import dataclass, field

import numpy as np

from wcode.exceptions.exceptions import ArrayMismatchException


@dataclass
class Perceptron:
    """
    The simplest form of a Neural Network
    """

    weigths: List[Number] | np.ndarray = field(default_factory=list)
    bias: Number = 0

    @classmethod
    def from_size(cls, weight_size: int, bias=0):
        """
        Todo: Create DocString
        """
        weights = [1]*weight_size
        return cls(weights, bias)

    def __post_init__(self):
        """
        Converts weights variable to numpy.ndarray if needed.
        """
        if not isinstance(self.weigths, np.ndarray):
            self.weigths = np.array(self.weigths).shape

    def predict(self, input_values: np.ndarray | list):
        """
        Todo: Create DocString
        """

        if isinstance(input_values, list):
            input_values = np.array(input_values)

        if input_values.shape != self.weigths.shape:
            raise ArrayMismatchException(input_values, self.weigths)

        return np.sum(input_values * self.weigths) + self.bias
