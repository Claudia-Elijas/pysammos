import numpy as np 

def test_function(a, b):
    if not (isinstance(a, (int, float, str)) and isinstance(b, type(a))):
        raise TypeError("test_function: Incompatible types in input: a and b must be int or float")
    return a + b




if __name__ == "__main__":
    a = 1
    b = 'A'
    print(test_function(a,b))
