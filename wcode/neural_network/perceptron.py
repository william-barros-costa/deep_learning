"""
    This module contains everything needed to create a single perceptron
"""
import numpy as np
from numbers import Number
from dataclasses import dataclass, field
from ..exceptions.exceptions import ArrayMismatchException
from typing import Union

@dataclass
class Perceptron:
    """
        The simplest form of a Neural Network
    """
    weigths: np.ndarray | list = field(default_factory=list)
    bias: Number = 0

    def __post_init__(self):
        print('Hello')
        if isinstance(self.weigths, list):
            self.weigths = np.array(self.weigths)
        

    def compute(self, X: np.array):
        if X.shape != self.weigths.shape:
            raise ArrayMismatchException(X, self.weigths)
        
        return np.sum(X * self.weigths) + self.bias

