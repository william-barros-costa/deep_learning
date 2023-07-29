import numpy as np

class ArrayMismatchException(Exception):
    """Exception raised for cases where two arrays are of different sizes.

    Attributes:
        array1 -- First Array
        array2 -- Second Array
        message -- explanation of the Exception
    """

    def __init__(self, array1: np.array, array2: np.array):
        self.array1 = array1
        self.array2 = array2
        self.message = f"Arrays have different dimensions. {self.array1.shape} != {self.array2.shape}"
        super().__init__(self.message)


class NoArgumentChosenException(Exception):
    """
        Exception raised for cases where user needs to select between two arguments but no selection was done

    Attributes:
        first_option -- The first available argument. Example ("name", <what was passed>, <what needs to be passed>)
        second_option -- The second available argument. Example ("name", <what was passed>, <what needs to be passed>)
        message -- Explanation of the Exception
    """


    def __init__(self, first_option: tuple, second_option: tuple) -> None:
        self.first_option_name, self.first_passed, self.first_needed = first_option
        self.second_option_name, self.second_passed, self.second_needed = second_option
        self.message = f"""No choice was done between arguments '{self.first_option_name}' and '{self.second_option_name}'
        This class is expecting you to pass one of the following options:
            - {self.first_option_name}: {self.first_needed}
            - {self.second_option_name}: {self.second_needed}
        Instead got:
            - {self.first_option_name}: {self.first_passed}
            - {self.second_option_name}: {self.second_passed}
        """
        super().__init__(self.message)


if __name__ == '__main__':
    array1 = np.zeros((1,2,10))
    array2 = np.zeros((1,3,11))

    raise ArrayMismatchException(array1, array2)


