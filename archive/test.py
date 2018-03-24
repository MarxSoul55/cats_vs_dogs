"""Testing out decorators."""

import time

import numpy as np


def timeit(function):
    """
    Times a function.

    # Decorations
        Prints out the seconds it took for the given function to run.
    """
    def wrapper(x):
        t0 = time.time()
        function(x)
        t1 = time.time()
        duration = t1 - t0
        print('The function took {} seconds to run.'.format(duration))
    return wrapper


@timeit
def f(x):
    for i in range(1000000):
        np.exp(x)


if __name__ == '__main__':
    f(2)
