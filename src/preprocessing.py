"""Provides interface for preprocessing image-related data."""

import os
import random
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image


class ImagePreprocessor:

    """Preprocesses images for a classifier."""

    def preprocess_image(self, path, rescale):
        """
        Given an image, grabs its pixels' RGB values as a tensor.
        Makes several modifications to that tensor and returns the result.

        # Parameters
            path (str): Path to the image. May be a URL.
            rescale (list, int): Width and height (columns and rows) of the resulting image.
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
            rescale (list): Width and height (columns and rows) that each image will be resized to.
                            ex: [1920, 1080]
        # Yields
            A tuple `(step, preprocessed_image_array, label_array)` starting from step 1.
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
