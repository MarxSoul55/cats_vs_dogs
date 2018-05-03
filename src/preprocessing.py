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

    def __init__(self, rescale, color_space):
        """
        # Instance Attributes
            self.rescale (list of two integers):
                - Desired [width, height] of the resulting tensor.
                - The interpolation algorithm used is bilinear interpolation.
            self.color_space (str):
                - Color space, or "representation", of the resulting tensor.
                - Options are: {'RGB', 'GRAYSCALE', 'RGB+GRAYSCALE', 'CIELAB', 'HSV'}
                - 'RGB' is simply the raw RGB values as given from the input tensor.
                  Output tensors will be HxWx3 in range [0, 1].
                - 'GRAYSCALE' is computed via OpenCV's implementation. See here:
                  https://bit.ly/2pUL2hR
                  Output tensors will be HxWx1 in range [0, 1].
                - 'RGB+GRAYSCALE' is simply RGB with a fourth channel—grayscale—as shown above.
                  Output tensors will be HxWx4 in range [0, 1].
                - 'CIELAB' is computed via OpenCV's implementation. See here:
                  https://bit.ly/2pUL2hR
                  'L' is bounded in [0, 1]; 'A' and 'B' are in [-1, 1].
                  The white reference point is from the D65 illuminant; shape is HxWx3.
                - 'HSV' is computed via OpenCV's implementation. See here:
                  https://bit.ly/2pUL2hR
                  Output tensors will be HxWx3 in range [0, 1].
        """
        self.rescale = rescale
        self.color_space = color_space

    def preprocess_image(self, path):
        """
        Given an image, grabs its pixels' RGB values as a tensor and converts it into a
        representation fitting the instance's attributes.

        # Parameters
            path (str):
                - Path to the image.
                - Can be a path to a local image on disk.
                - May also be a URL that returns the image by itself.
        # Returns
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

    def preprocess_directory(self, path):
        """
        An extension of `ImagePreprocessor.preprocess_image` for directories.
        Given a directory, preprocesses images in it with `ImagePreprocessor.preprocess_image`.
        Subdirectories and files of unsupported formats are ignored.

        # Parameters
            path (str):
                - Path to the directory.
        # Yields
            A list `[filename, preprocessed_image_array]`.
            See `ImagePreprocessor.preprocess_image` for details on the latter.
        """
        path = os.path.abspath(path)
        for objectname in os.listdir(path):
            extension = os.path.splitext(os.path.join(path, objectname))[1].lower()
            if extension not in self.SUPPORTED_FORMATS:
                continue
            image_path = os.path.join(path, objectname)
            preprocessed_image = self.preprocess_image(image_path)
            yield objectname, preprocessed_image

    def preprocess_classes(self, steps, train_dir, encoding):
        """
        Given a directory of subdirectories of images, preprocesses an image from the 1st subdir,
        then the 2nd, then the Nth, and then loops back towards the 1st and gets another image,
        etc. The order of the images in each subdir is randomized. After all images in a directory
        have been preprocessed (given that `steps` is big enough), preprocessing will start over at
        the beginning of the directory in question. The order of images won't be rerandomized.

        # Parameters
            steps (int):
                - Amount of step-input-label triplets to generate.
            train_dir (str):
                - Path to the directory of classes.
                - May be relative or absolute.
                - e.g. 'data/cats/train' (where 'train' holds the subdirs)
            encoding (dict):
                - Maps the name of the subdirectory (class) to a label.
                - ex: {'cats': [1, 0], 'dogs': [0, 1]}
        # Yields
            A tuple `(step, preprocessed_image_array, label_array)` starting from step 1.
        """
        classes = os.listdir(train_dir)
        cursors = {}
        images = {}
        for class_ in classes:
            cursors[class_] = 0
            images[class_] = os.listdir(os.path.join(train_dir, class_))
            random.shuffle(images[class_])
        # I know what you're thinking, but this CANNOT be implemented with a double-for-loop.
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
