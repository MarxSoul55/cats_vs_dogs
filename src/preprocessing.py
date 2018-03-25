"""Provides interface for preprocessing image-related data."""

import os
import random

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
        Given a directory of subdirectories of images, preprocesses an image from the 1st subdir,
        then the 2nd, then the Nth, and then loops back towards the 1st and gets another image,
        etc. The order of the images in each subdir is randomized. After all images in a directory
        have been preprocessed (given that `steps` is big enough), preprocessing will start over at
        the beginning of the directory in question. The order of images won't be randomized again.

        # Parameters
            steps (int): Amount of step-input-label triplets to generate (aka amount of images that
                         will be preprocessed).
            train_dir (str): Path to the directory of classes. May be relative or absolute.
            encoding (dict): Maps the name of the subdirectory (class) to a label.
                             ex: {'cats': [1, 0], 'dogs': [0, 1]}
            rescale (list): Width and height that each image will be resized to.
                            ex: [1920, 1080]
        # Yields
            A tuple (step, preprocessed_image_array, label_array) starting from step 1.
        """
        train_dir = os.path.abspath(train_dir)
        class_names = os.listdir(train_dir)
        class_paths = [os.path.join(train_dir, class_name) for class_name in class_names]
        cursors = {}
        images = {}
        for class_name, class_path in zip(class_names, class_paths):
            cursors[class_name] = 0
            images[class_name] = os.listdir(class_path)
            random.shuffle(images[class_name])
        step = 0
        while True:
            for class_name, class_path in zip(class_names, class_paths):
                if step < steps:
                    step += 1
                else:
                    return
                image_path = os.path.join(class_path, images[class_name][cursors[class_name]])
                preprocessed_image = self.preprocess_image(image_path, rescale)
                preprocessed_image = np.array([preprocessed_image])
                label = np.array([encoding[class_name]]).astype('float32')
                if cursors[class_name] == (len(images[class_name]) - 1):
                    cursors[class_name] = 0
                else:
                    cursors[class_name] += 1
                yield step, preprocessed_image, label
