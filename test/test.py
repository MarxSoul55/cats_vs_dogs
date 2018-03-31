"""Tests the runtime of specified functions"""

import time

import cv2
import numpy as np


def timeit(function):
    """
    Times a function.

    # Decorations
        Prints out the seconds it took for the given function to run.
    """
    def wrapper(*args, **kwargs):
        t0 = time.time()
        function(*args, **kwargs)
        t1 = time.time()
        duration = t1 - t0
        print('{} took {} seconds to run.'.format(function.__name__, duration))
    return wrapper


@timeit
def test1(path, rescale):
        image = cv2.imread(path)
        preprocessed_image = cv2.resize(image, tuple(rescale), interpolation=cv2.INTER_NEAREST)
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
        return preprocessed_image


@timeit
def test2(path, rescale):
    image = cv2.imread(path)
    preprocessed_image = cv2.resize(image, tuple(rescale), interpolation=cv2.INTER_NEAREST)
    preprocessed_image = np.average(preprocessed_image, axis=2)
    return preprocessed_image


if __name__ == '__main__':
    abs_path = 's:/mystuff/projects/cv/cats_vs_dogs/src/data/test_small/1.jpg'
    for trial in range(1, 6):
        print('TRIAL {}'.format(trial))
        test1(abs_path, [256, 256])
    print('-----------------------------------------------------------')
    for trial in range(1, 6):
        print('TRIAL {}'.format(trial))
        test2(abs_path, [256, 256])
