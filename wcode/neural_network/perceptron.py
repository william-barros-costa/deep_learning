"""
    This module contains everything needed to create a single perceptron
"""
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Perceptron:
    """
        The simplest form of a Neural Network
    """
    weigths: np.array = field(default_factory=lambda : np.array([]))

    def compute(self):
        """

        """
        pass

