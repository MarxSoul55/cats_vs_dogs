"""Provides class which implements a customizable pipeline."""

import os
import random

import cv2
import numpy as np

import imops


class ImageDataPipeline:

    """Specifies a custom pipeline object which performs a sequence of image preprocessing ops."""

    def __init__(self,
                 rescale=[128, 128],
                 colorspace='CIELAB',
                 current_bounds=[[0, 255], [0, 255], [0, 255]],
                 desired_bounds=[[0, 1], [-1, 1], [-1, 1]],
                 dtype='float32'):
        """
        Instance Attributes:
            - For each attribute, see the function(s) listed in the imops module for details.
                - rescale
                    - imops.resize_image
                - colorspace
                    - imops.convert_colorspace
                - current_bounds, desired_bounds, dtype
                    - imops.normalize_image
        """
        self.rescale = rescale
        self.colorspace = colorspace
        self.current_bounds = current_bounds
        self.desired_bounds = desired_bounds
        self.dtype = dtype

    def valid_file(self,
                   path):
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
                - Datatype is uint8.
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
                - Datatype is uint8.
            - rescale (list of two ints)
                - Desired [height, width].
                - e.g. [1080, 1920]
        Returns:
            - An idential tensor, with a different height/width as per rescale arg.
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
                - Datatype is uint8.
            - colorspace (str)
                - Options are: 'RGB' (unchanged), 'GRAYSCALE', 'RGB+GRAYSCALE', 'CIELAB', 'HSV'
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
        if colorspace == 'CIELAB':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif colorspace == 'GRAYSCALE':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif colorspace == 'RGB+GRAYSCALE':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.dstack((image, gray))
        elif colorspace == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return image

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

    def preprocess_image(self,
                         path):
        """
        Pipeline for complete preprocessing of an image.
        Loads image, rescales it, converts colorspace, normalizes, and converts datatype.
        Finalizes by converting HWC to NHWC.

        Parameters:
            - path (str)
                - Path to the image to preprocess.
        Returns:
            - The fully preprocessed image in NHWC format.
        """
        image = imops.load_image(path)
        image = imops.resize_image(image, self.rescale)
        image = imops.convert_colorspace(image, self.colorspace)
        image = imops.normalize_image(image, self.current_bounds, self.desired_bounds,
                                      dtype=self.dtype)
        image = np.expand_dims(image, axis=0)
        return image

    def preprocess_directory(self,
                             path):
        """
        An extension of the preprocess_image method for directories.
        Given a directory, preprocesses images in it with the preprocess_image method.
        Subdirectories and files of unsupported formats are ignored.

        Parameters:
            - path (str)
                - Path to the directory.
        Yields:
            - A list [filename, preprocessed_image_array].
                - See the preprocess_image function in this module for details on the latter.
        """
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if not imops.valid_file(filepath):
                continue
            preprocessed_image = self.preprocess_image(filepath)
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
            - encoding (dict, str --> np.ndarray)
                - Maps the name of the subdirectory (class) to a label.
                    - e.g. {'cats': np.array([[1, 0]]), 'dogs': np.array([[0, 1]])}
                        - Each label must have the same shape!
                        - In this case, the two labels are of shape [1, 2].
        Yields:
            - A tuple (step, image_path, preprocessed_image_array, label_array) starting w/ step 1.
        """
        classes = os.listdir(train_dir)
        cursors = {}
        images = {}
        for class_ in classes:
            cursors[class_] = 0
            images[class_] = os.listdir(os.path.join(train_dir, class_))
            random.shuffle(images[class_])
        step = 0
        while True:
            for class_ in classes:
                if step < steps:
                    step += 1
                else:
                    return
                image_path = os.path.join(train_dir, class_, images[class_][cursors[class_]])
                if not imops.valid_file(image_path):
                    continue
                preprocessed_image = self.preprocess_image(image_path)
                label = encoding[class_]
                if cursors[class_] == (len(images[class_]) - 1):
                    cursors[class_] = 0
                else:
                    cursors[class_] += 1
        yield step, image_path, preprocessed_image, label
