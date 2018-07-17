"""Provides interface for preprocessing image-related data."""

import os
import random

import cv2
import numpy as np


class ImagePreprocessor:

    """Encapsulates methods for loading, editing, and returning images as tensors."""

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

    def valid_file(self,
                   path):
        """
        Ensures path points to a file of a supported filetype.

        Parameters:
            - path (str)
                - Path to the file.
        Returns:
            - A boolean; true if a file of a supported filetype; false if not.
        """
        extension = os.path.splitext(path)[1].lower()
        if os.path.isfile(path) and extension in self.SUPPORTED_FORMATS:
            return True
        return False

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
        Normalizes an image CHANNELWISE.
        Changes the boundaries of the interval in which the image's numerical values lie.

        Parameters:
            - image (tensor)
                - Formatted in HWC.
                - Datatype is `uint8`.
            - current_bounds (list of lists of two ints each)
                - e.g. For a `uint8` image with 2 channels: [[0, 255], [0, 255]]
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

    def preprocess_image(self,
                         path,
                         rescale,
                         colorspace,
                         current_bounds,
                         desired_bounds,
                         dtype='float32'):
        """
        Pipeline for complete preprocessing of an image.
        Loads image, rescales it, converts colorspace, normalizes, and converts datatype.

        Parameters:
            - See the following methods for details for each parameter in order.
                - `load_image`
                - `resize_image`
                - `convert_colorspace`
                - `normalize_image`
        Returns:
            - The fully preprocessed image.
        """
        image = self.load_image(path)
        image = self.resize_image(image, rescale)
        image = self.convert_colorspace(image, colorspace)
        image = self.normalize_image(image, current_bounds, desired_bounds, dtype=dtype)
        return image

    def preprocess_directory(self,
                             path,
                             rescale,
                             colorspace,
                             current_bounds,
                             desired_bounds,
                             dtype='float32'):
        """
        An extension of `ImagePreprocessor.preprocess_image` for directories.
        Given a directory, preprocesses images in it with `ImagePreprocessor.preprocess_image`.
        Subdirectories and files of unsupported formats are ignored.

        Parameters:
            - path (str)
                - Path to the directory.
            - For other parameters, see the `preprocess_image` method for details.
        Yields:
            - A list `[filename, preprocessed_image_array]`.
                - See `ImagePreprocessor.preprocess_image` for details on the latter.
        """
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            extension = os.path.splitext(filepath)[1].lower()
            if extension not in self.SUPPORTED_FORMATS:
                continue
            preprocessed_image = self.preprocess_image(filepath, rescale, colorspace,
                                                       current_bounds, desired_bounds, dtype=dtype)
            yield filename, preprocessed_image

    def preprocess_classes(self,
                           steps,
                           train_dir,
                           encoding):
        """
        Given a directory of subdirectories of images, preprocesses an image from the 1st subdir,
        then the 2nd, then the Nth, and then loops back towards the 1st and gets another image,
        etc. The order of the images in each subdir is randomized. After all images in a subdir
        have been preprocessed (given that `steps` is big enough), preprocessing will start over at
        the beginning of the subdir in question. The order of images within each subdir is
        randomized at the start, but not randomized again afterwards.

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
