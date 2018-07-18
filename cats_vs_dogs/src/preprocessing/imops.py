"""A collection of functions for loading and editing images as tensors."""

import os

import cv2
import numpy as np


def valid_file(path):
    """
    Ensures path points to a file of a supported filetype by OpenCV.

    Parameters:
        - path (str)
            - Path to the file.
    Returns:
        - A boolean; true if valid, false if not.
    """
    supported_formats = [
        '.bmp',
        '.pbm',
        '.pgm',
        '.ppm',
        '.sr',
        '.ras',
        '.jpeg',
        '.jpg',
        '.jpe',
        '.jp2',
        '.tiff',
        '.tif',
        '.png'
    ]
    extension = os.path.splitext(path)[1].lower()
    if extension in supported_formats:
        return True
    return False


def load_image(path):
    """
    Loads an image's RGB values as a tensor.

    Parameters:
        - path (str)
            - Path to the image.
    Returns:
        - A tensor.
            - Encoded in RGB.
            - Formatted in HWC.
            - Datatype is uint8.
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_image(image,
                 rescale):
    """
    Resizes an image with bilinear interpolation.

    Parameters:
        - image (tensor)
            - Encoded in RGB.
            - Formatted in HWC.
            - Datatype is uint8.
        - rescale (list of two ints)
            - Desired [height, width].
            - e.g. [1080, 1920]
    Returns:
        - An idential tensor, with a different height/width as per rescale arg.
    """
    return cv2.resize(image, tuple(reversed(rescale)), interpolation=cv2.INTER_LINEAR)


def convert_colorspace(image,
                       colorspace):
    """
    Converts an RGB image to a different color space.

    Parameters:
        - image (tensor)
            - Encoded in RGB.
            - Formatted in HWC.
            - Datatype is uint8.
        - colorspace (str)
            - Options are: 'GRAYSCALE', 'RGB+GRAYSCALE', 'CIELAB', 'HSV'
                - 'GRAYSCALE' is computed via OpenCV's implementation.
                    - https://bit.ly/2pUL2hR
                    - Output tensors will be HxWx1 in range [0, 1].
                - 'RGB+GRAYSCALE' is simply RGB with a fourth channel—grayscale.
                    - Output tensors will be HxWx4 in range [0, 1].
                    - Grayscale is computed as per the same link above.
                - 'CIELAB' is computed via OpenCV's implementation.
                    - https://bit.ly/2pUL2hR
                    - 'L' is bounded in [0, 1]; 'A' and 'B' are in [-1, 1].
                    - The white reference point is from the D65 illuminant; shape is HxWx3.
                - 'HSV' is computed via OpenCV's implementation.
                    - https://bit.ly/2pUL2hR
                    - Output tensors will be HxWx3 in range [0, 1].
    Returns:
        - The converted tensor.
        - Still in uint8.
        - Still formatted in HWC, but may have different number of channels.
    """
    if colorspace == 'GRAYSCALE':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    elif colorspace == 'RGB+GRAYSCALE':
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.dstack((image, gray))
        return image
    elif colorspace == 'CIELAB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return image
    elif colorspace == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return image


def normalize_image(image,
                    current_bounds,
                    desired_bounds,
                    dtype='float32'):
    """
    Normalizes an image CHANNELWISE.
    Changes the boundaries of the interval in which the image's numerical values lie.

    Parameters:
        - image (tensor)
            - Formatted in HWC.
            - Datatype is uint8.
        - current_bounds (list of lists of two ints each)
            - e.g. For a uint8 image with 2 channels: [[0, 255], [0, 255]]
        - desired_bounds (list of lists of two ints each)
            - The desired boundaries for the new tensor.
        - dtype (str)
            - A numpy-compatible datatype.
    Returns:
        - The resulting tensor.
        - Still formatted in HWC.
        - Only difference is the datatype and range of allowed numbers.
    """
    image = image.astype(dtype)
    number_of_channels = image.shape[2]
    for channel in range(0, number_of_channels):
        image[:, :, channel] += -current_bounds[channel][0]
        image[:, :, channel] /= (current_bounds[channel][1] /
                                 (desired_bounds[channel][1] - desired_bounds[channel][0]))
        image[:, :, channel] += desired_bounds[channel][0]
    return image
