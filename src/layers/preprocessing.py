"""Provides interface for preprocessing data."""

import os

import cv2
import numpy as np


class ImagePreprocessor:

    """Helps preprocess images for a classifier."""

    def preprocess_image(self, image, rescale):
        """
        Preprocesses an image for the model.
        Converts image to a 256x256x3, 8-bit LAB representation.

        # Parameters
            image (str): Path to the image.
            rescale (tuple): (width, height) of the resulting image.
        # Returns
            A preprocessed image as a numpy-array.
        """
        image = cv2.imread(image)
        image = cv2.resize(image, rescale, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        if image.dtype != 'uint8':
            raise TypeError('When preprocessing `{}`, expected `uint8`, but got `{}`.'
                            .format(image, image.dtype))
        image = image.astype('float32')
        image /= 255
        return image

    def preprocess_directory(self, steps, train_dir, encoding, rescale):
        """
        Given a directory of subdirectories of images, loops through the subdirectories,
        preprocessing one image from each then moving to the next subdirectory,
        and loops back when the last subdirectory is passed.
        Preprocesses the image using the `preprocess_image` method,
        and generates a binary-encoded label based off of `encoding`.

        # Parameters
            steps (int): Amount of data-label pairs to generate.
            train_dir (str): Path to the directory of classes.
            encoding (dict, str -> list): Maps the name of the subdirectory (class) to a
                                          binary-encoded label.
            rescale (tuple): (width, height) that each image will be resized to.
        # Yields
            `(step, data_array, label_array)` starting from step 1.
        """
        classes = os.listdir(train_dir)
        cursors = {}
        images = {}
        for class_ in classes:
            cursors[class_] = 0
            images[class_] = sorted(os.listdir(train_dir + '/' + class_))
        step = 0
        while True:
            for class_ in classes:
                if step < steps:
                    step += 1
                else:
                    return
                absolute_path = os.path.abspath(train_dir + '/' + class_ + '/' +
                                                images[class_][cursors[class_]])
                preprocessed = self.preprocess_image(absolute_path, rescale)
                data = np.array([preprocessed])
                label = np.array([encoding[class_]]).astype('float32')
                if cursors[class_] == (len(images[class_]) - 1):
                    cursors[class_] = 0
                else:
                    cursors[class_] += 1
                yield step, data, label
