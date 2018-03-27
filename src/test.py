"""Tests the runtime of specified functions"""

import os
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image


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
def preprocess_image(path, rescale):
        """
        Given an image, grabs its pixels' RGB values as a tensor.
        Makes several modifications to that tensor and returns the result.

        # Parameters
            path (str): Path to the image. May be a URL.
            rescale (list): Width and height (columns and rows) of the resulting tensor.
                            ex: [1920, 1080]
        # Returns
            A numpy array with shape `rescale[0] X rescale[1] X 3` (width X height X channels).
            The 3 channels are that of CIELAB, which are L -> A -> B in that order of indices.
            They are 'float32' values in range [-1, 1] for all 3 channels.
        # Raises
            TypeError: if the image's bit depth isn't 24.
        """
        if os.path.exists(path):
            image = cv2.imread(path)
        else:  # If the path doesn't exist on disk, it must be a URL.
            response = requests.get(path)
            pil_object = Image.open(BytesIO(response.content))
            image = np.array(pil_object)
        if image.dtype != 'uint8':
            raise TypeError('When preprocessing `{}`, expected `uint8`, but got `{}`.'
                            .format(image, image.dtype))
        preprocessed_image = cv2.resize(image, tuple(rescale),
                                        interpolation=cv2.INTER_NEAREST)
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2LAB)
        preprocessed_image = preprocessed_image.astype('float32')
        preprocessed_image /= 255
        preprocessed_image *= 2
        preprocessed_image -= 1
        return preprocessed_image


@timeit
def preprocess_image_different(path, rescale):
        """
        Given an image, grabs its pixels' RGB values as a tensor.
        Makes several modifications to that tensor and returns the result.

        # Parameters
            path (str): Path to the image. May be a URL.
            rescale (list): Width and height (columns and rows) of the resulting tensor.
                            ex: [1920, 1080]
        # Returns
            A numpy array with shape `rescale[0] X rescale[1] X 3` (width X height X channels).
            The 3 channels are that of CIELAB, which are L -> A -> B in that order of indices.
            They are 'float32' values in range [-1, 1] for all 3 channels.
        # Raises
            TypeError: if the image's bit depth isn't 24.
        """
        if os.path.exists(path):
            image = cv2.imread(path)
        else:  # If the path doesn't exist on disk, it must be a URL.
            response = requests.get(path)
            pil_object = Image.open(BytesIO(response.content))
            image = np.array(pil_object)
        if image.dtype != 'uint8':
            raise TypeError('When preprocessing `{}`, expected `uint8`, but got `{}`.'
                            .format(image, image.dtype))
        preprocessed_image = cv2.resize(image, tuple(rescale),
                                        interpolation=cv2.INTER_NEAREST)
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2LAB)
        preprocessed_image = preprocessed_image.astype('float32')
        preprocessed_image = ((preprocessed_image / 255) * 2) - 1
        return preprocessed_image


if __name__ == '__main__':
    for _ in range(10):
        preprocess_image_different('data/test_small/1.jpg', [256, 256])
        preprocess_image('data/test_small/1.jpg', [256, 256])
