"""
    This module contains everything needed to create a single perceptron
"""
import numpy as np
from numbers import Number
from dataclasses import dataclass, field
from ..exceptions.exceptions import ArrayMismatchException
from typing import List

@dataclass
class Perceptron:
    """
        The simplest form of a Neural Network
    """
    weigths: List[Number] = field(default_factory=list)
    bias: Number = 0

    def __post_init__(self):
        """
            Converts weights variable to numpy.ndarray if needed.
        """

        if not isinstance(self.weigths, np.ndarray):
            self.weigths = np.array(self.weigths)
        

    def predict(self, X: np.ndarray | list):
        if isinstance(X, list):
            X= np.array(X)

        if X.shape != self.weigths.shape:
            raise ArrayMismatchException(X, self.weigths)
        
        return np.sum(X * self.weigths) + self.bias

