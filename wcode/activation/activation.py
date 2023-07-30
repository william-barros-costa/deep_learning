"""
            Todo: Create DocString
"""
from dataclasses import dataclass
from typing import Callable
from numbers import Number

@dataclass
class Activation:
    """
    A class to represent an Activation Function.

    ...

    Attributes
    ----------
    activation_function: Callable[Number] -> Number
        - Activation function used to calculate the final value

    Methods
    -------
    compute(value:Number):
        - Applies an activation function to a value and returns its result.

    """
    activation_function: Callable[[Number], Number] = lambda x: x

    def compute(self, value: Number) -> Number:
        """
        Applies an activation function to a value and returns its result.

            Parameters:
                value (Number): Any number type

            Returns:
                activated_value (Number): Number resulting from the function activation_function

        """
        return self.activation_function(value)
