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
        image = cv2.resize(image, rescale)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = image.astype('float32')
        image /= 255
        return image

    def preprocess_directory(self, steps, train_dir, order, rescale=(256, 256)):
        """
        Given a directory of subdirectories of classes, preprocesses their contents and yields
        the results along with one-hot labels.

        # Parameters
            steps (int): Preprocess `steps` amount of samples from each class.
            train_dir (str): Path to the directory of classes.
            order (list): Classes; order determines the one-hot label.
            rescale (tuple): (width, height) that each image will be resized to.
        # Yields
            Current step (starting from 1) and a data-label pair.
        """
        classes = os.listdir(train_dir)
        cursors = {}
        images = {}
        for class_ in classes:
            cursors[class_] = 0
            images[class_] = sorted(os.listdir(train_dir + '/' + class_))
        for step in range(steps):
            for class_ in classes:
                absolute_path = os.path.abspath(train_dir + '/' + class_ + '/' +
                                                images[class_][cursors[class_]])
                preprocessed = self.preprocess_image(absolute_path, rescale)
                data = np.array([preprocessed])
                label = np.array([np.zeros(len(classes))])
                label[0][order.index(class_)] = 1
                label = label.astype('float32')
                if cursors[class_] == (len(images[class_]) - 1):
                    cursors[class_] = 0
                else:
                    cursors[class_] += 1
                yield step + 1, data, label
