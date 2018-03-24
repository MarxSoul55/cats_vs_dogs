"""Testing out decorators."""

import time


def timeit(function):
    def wrapper():
        t0 = time.time()
        function()
        t1 = time.time()
        duration = t1 - t0
        print('The function took {} seconds to run.'.format(duration))
    return wrapper


@timeit
def sqrt(x):
    return x ** 0.5


if __name__ == '__main__':
    sqrt(2)
