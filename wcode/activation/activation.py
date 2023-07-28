import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from numbers import Number

@dataclass
class Activation:
    
    activation_function: Callable[[Number], Number] = lambda x: x 

    def compute(self, x: Number) -> Number:
        return self.activation_function(x)