from dataclasses import dataclass
from typing import Callable, List
from numbers import Number
import numpy as np


@dataclass
class Layer:
    layer_function: Callable[[List[Number]], List[Number]] = lambda x: x

    def predict(self, X: List[Number]):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        return self.layer_function(X)
