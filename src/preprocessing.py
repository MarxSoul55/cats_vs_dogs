"""Provides interface for preprocessing image-related data."""

import os

import cv2
import numpy as np


class ImagePreprocessor:

    """Preprocesses images for a classifier."""

    def preprocess_image(self, path, rescale):
        """
        Preprocesses an image into a tensor representation.

        # Parameters
            path (str): Path to the image.
            rescale (list, int): Desired [columns, rows] of the resulting image.
        # Returns
            A preprocessed image as a numpy-array whose shape is determined by `rescale`.
            Specifically, a CIELAB (D65) representation in 'float32' in range [-1, 1].
        # Raises
            TypeError: if the image's bit depth isn't 24.
        """
        image = cv2.imread(path)
        image = cv2.resize(image, tuple(rescale), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        if image.dtype != 'uint8':
            raise TypeError('When preprocessing `{}`, expected `uint8`, but got `{}`.'
                            .format(image, image.dtype))
        image = image.astype('float32')
        image /= 255
        image *= 2
        image -= 1
        return image

    def preprocess_classes(self, steps, train_dir, encoding, rescale):
        """
        Given a directory of subdirectories of images, where each subdirectory is a "class"...
        preprocesses one image from each subdirectory and moves onto the next...
        when reaching the last subdirectory, loops back to the first.
        Uses the `preprocess_image` method.
        The label is converted from a list (`encoding[subdirectory]`) to a numpy-array.

        # Parameters
            steps (int): Amount of data-label pairs to generate.
            train_dir (str): Path to the directory of classes.
            encoding (dict, str -> list): Maps the name of the subdirectory (class) to a label.
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
