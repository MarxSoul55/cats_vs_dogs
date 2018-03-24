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
        etc. The order of the images in each subdir is randomized. After all images have been
        preprocessed (given that `steps` is big enough), the subdirs are shuffled again and the
        preprocessing continues.

        # Parameters
            steps (int): Amount of step-input-label triplets to generate (aka amount of images that
                         will be preprocessed).
            train_dir (str): Path to the directory of classes. May be relative or absolute.
            encoding (dict): Maps the name of the subdirectory (class) to a vector.
                             ex: {'cats': [1, 0], 'dogs': [0, 1]}
            rescale (list): Width and height that each image will be resized to.
                            ex: [1920, 1080]
        # Yields
            A tuple (step, preprocessed_image_array, label_array) starting from step 1.
        """
        # TWO BIG PROBLEMS WITH THIS CODE
        # 1. Steps are duplicated since the same step goes for multiple classes
        # 2. Some steps are useless because they're spend refreshing `image_paths`
        train_dir = os.path.abspath(train_dir)
        class_names = os.listdir(train_dir)
        class_paths = [os.path.join(train_dir, name) for name in class_names]
        image_paths = {class_name: [] for class_name in class_names}
        for step in range(1, steps + 1):
            for class_name, class_path in zip(class_names, class_paths):
                if len(image_paths[class_name]) == 0:
                    image_names = os.listdir(class_path)
                    random.shuffle(image_names)
                    image_paths[class_name] = [os.path.join(class_path, image_name)
                                               for image_name in image_names]
                    continue
                # Else, since we still have images left, proceed as normal!
                input_ = image_paths[class_name].pop()
                input_ = self.preprocess_image(input_, rescale)
                input_ = np.array([input_])
                label = np.array([encoding[class_name]]).astype('float32')
                yield step, input_, label
