from ..exceptions.exceptions import NoArgumentChosenException
from ..neural_network.perceptron import Perceptron
from dataclasses import dataclass, field
from numbers import Number
from typing import List
import numpy as np

@dataclass
class Dense:

    shape: Number = -1
    neurons: List[Perceptron] = field(default_factory=list)

    def __post_init__(self):
        if len(self.neurons) == 0:
            if self.shape <= 0:
                raise NoArgumentChosenException(
                    first_option=("Number of Neurons", self.shape, "Integer above 0"),
                    second_option=("Neurons", self.neurons, "List of Perceptron"),
                )
            self.neurons = [Perceptron() for _ in range(self.shape)]
        self.shape = len(self.neurons)
