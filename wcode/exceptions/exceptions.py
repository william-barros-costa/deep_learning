import numpy as np

class ArrayMismatchException(Exception):
    """Exception raised for cases where two arrays are of different sizes.

    Attributes:
        array1 -- First Array
        array2 -- Second Array
        message -- explanation of the error
    """

    def __init__(self, array1: np.array, array2: np.array):
        self.array1 = array1
        self.array2 = array2
        self.message = f"Arrays have different dimensions. {self.array1.shape} != {self.array2.shape}"
        super().__init__(self.message)


if __name__ == '__main__':
    array1 = np.zeros((1,2,10))
    array2 = np.zeros((1,3,11))

    raise ArrayMismatchException(array1, array2)