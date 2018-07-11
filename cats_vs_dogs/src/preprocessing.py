"""Provides interface for preprocessing image-related data."""

import os
import random
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image


class ImagePreprocessor:

    """A preprocessor that handles visual data encoded in various image formats."""

    SUPPORTED_FORMATS = [
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

    def __init__(self,
                 rescale,
                 color_space):
        """
        Instance Attributes:
            - self.rescale (list of two integers)
                - Desired [width, height] of the resulting tensor.
                - The interpolation algorithm used is bilinear interpolation.
            - self.color_space (str)
                - Options are: {'RGB', 'GRAYSCALE', 'RGB+GRAYSCALE', 'CIELAB', 'HSV'}
                    - 'RGB' is simply the raw RGB values as given from the input tensor.
                        - Output tensors will be HxWx3 in range [0, 1].
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
        """
        self.rescale = rescale
        self.color_space = color_space

    def load_image(self,
                   path):
        """
        Loads an image's RGB values as a tensor.

        Parameters:
            - path (str)
                - Path to the image.
        Returns:
            - A tensor.
                - Encoded in RGB.
                - Formatted in HWC.
                - Datatype is `uint8`.
        """
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def resize_image(self,
                     image,
                     rescale):
        """
        Resizes an image with bilinear interpolation.

        Parameters:
            - image (tensor)
                - Encoded in RGB.
                - Formatted in HWC.
                - Datatype is `uint8`.
            - rescale (list of two ints)
                - Desired `[height, width]`.
                - e.g. [1080, 1920]
        Returns:
            - An idential tensor, with a different height/width as per `rescale`.
        """
        return cv2.resize(image, tuple(reversed(rescale)), interpolation=cv2.INTER_LINEAR)

    def convert_colorspace(self,
                           image,
                           colorspace):
        """
        Converts an RGB image to a different color space.

        Parameters:
            - image (tensor)
                - Encoded in RGB.
                - Formatted in HWC.
                - Datatype is `uint8`.
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
            - Still in `uint8`.
            - Still formatted in HWC, but may have different number of channels.
        Raises:
            - ValueError
                - ...if an invalid argument is given for `colorspace`.
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
        else:
            raise ValueError('Invalid argument for parameter `colorspace`.')

    def normalize_image(self,
                        image,
                        current_bounds,
                        desired_bounds,
                        dtype='float32'):
        """
        Changes the boundaries of the interval in which the image's numerical values lie.
        e.g. Converting `uint8` in [0, 255] to `float32` in [0, 1].

        Parameters:
            - image (tensor)
                - Datatype is `uint8`.
            - current_bounds (list of two ints)
                - Lower, then upper boundary.
                - e.g. The image might be in `uint8`, so current_bounds=[0, 255].
            - desired_bounds (list of two ints)
                - The desired boundaries for the new tensor.
            - dtype (str)
                - A numpy-compatible datatype.
        Returns:
            - The resulting tensor.
            - Still formatted in HWC.
            - Only difference is the datatype and range of allowed numbers.
        """
        image = image.astype(dtype)
        image += -current_bounds[0]
        image /= (current_bounds[1] / (desired_bounds[1] - desired_bounds[0]))
        image += desired_bounds[0]
        return image

    def preprocess_image(self,
                         path):
        """
        Given an image, grabs its pixels' RGB values as a tensor and converts it into a
        representation fitting the instance's attributes.

        Parameters:
            - path (str)
                - Path to the image.
                - Can be a path to a local image on disk.
                - May also be a URL that returns the image by itself.
        Returns:
            - A numpy array, customized according to the instance's attributes.
            - Type will be 'float32'.
        """
        if os.path.exists(path):
            image = cv2.imread(path)
        else:
            response = requests.get(path)
            pil_object = Image.open(BytesIO(response.content))
            image = np.array(pil_object)
        image = cv2.resize(image, tuple(self.rescale), interpolation=cv2.INTER_LINEAR)
        if self.color_space == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
            image /= 255
            return image
        elif self.color_space == 'GRAYSCALE':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float32')
            image /= 255
            image = np.expand_dims(image, axis=2)
            return image
        elif self.color_space == 'RGB+GRAYSCALE':
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float32')
            image = np.dstack((rgb, gray))
            image /= 255
            return image
        elif self.color_space == 'CIELAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype('float32')
            image[:, :, 0] /= 255
            image[:, :, 1] = ((image[:, :, 1] / 255) * 2) - 1
            image[:, :, 2] = ((image[:, :, 2] / 255) * 2) - 1
            return image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float32')
            image /= 255
            return image

    def preprocess_directory(self,
                             path):
        """
        An extension of `ImagePreprocessor.preprocess_image` for directories.
        Given a directory, preprocesses images in it with `ImagePreprocessor.preprocess_image`.
        Subdirectories and files of unsupported formats are ignored.

        Parameters:
            - path (str)
                - Path to the directory.
        Yields:
            - A list `[filename, preprocessed_image_array]`.
                - See `ImagePreprocessor.preprocess_image` for details on the latter.
        """
        path = os.path.abspath(path)
        for objectname in os.listdir(path):
            extension = os.path.splitext(os.path.join(path, objectname))[1].lower()
            if extension not in self.SUPPORTED_FORMATS:
                continue
            image_path = os.path.join(path, objectname)
            preprocessed_image = self.preprocess_image(image_path)
            yield objectname, preprocessed_image

    def preprocess_classes(self,
                           steps,
                           train_dir,
                           encoding):
        """
        Given a directory of subdirectories of images, preprocesses an image from the 1st subdir,
        then the 2nd, then the Nth, and then loops back towards the 1st and gets another image,
        etc. The order of the images in each subdir is randomized. After all images in a directory
        have been preprocessed (given that `steps` is big enough), preprocessing will start over at
        the beginning of the directory in question. The order of images won't be rerandomized.

        Parameters:
            - steps (int)
                - Amount of step-input-label triplets to generate.
            - train_dir (str)
                - Path to the directory of classes.
                - May be relative or absolute.
                - e.g. 'data/cats/train' (where 'train' holds the subdirs)
            - encoding (dict)
                - Maps the name of the subdirectory (class) to a label.
                - ex: {'cats': [1, 0], 'dogs': [0, 1]}
        Yields:
            - A tuple `(step, preprocessed_image_array, label_array)` starting from step 1.
        """
        classes = os.listdir(train_dir)
        cursors = {}
        images = {}
        for class_ in classes:
            cursors[class_] = 0
            images[class_] = os.listdir(os.path.join(train_dir, class_))
            random.shuffle(images[class_])
        # I know what you're thinking, but this CANNOT be implemented with a double-for-loop.
        # At least, not nicely. Do prove me wrong!
        step = 0
        while True:
            for class_ in classes:
                if step < steps:
                    step += 1
                else:
                    return
                image_path = os.path.join(train_dir, class_, images[class_][cursors[class_]])
                extension = os.path.splitext(image_path)[1].lower()
                if extension not in self.SUPPORTED_FORMATS:
                    continue
                preprocessed_image = self.preprocess_image(image_path)
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
                label = np.expand_dims(encoding[class_], axis=0).astype('float32')
                if cursors[class_] == (len(images[class_]) - 1):
                    cursors[class_] = 0
                else:
                    cursors[class_] += 1
                yield step, preprocessed_image, label