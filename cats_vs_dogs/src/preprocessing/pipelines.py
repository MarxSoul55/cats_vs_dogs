"""Pipelines which perform complete image preprocessing."""

import os
import random

import numpy as np

import imops


class ImageDataPipeline:

    """Specifies a pipeline object which performs multiple image preprocessing steps."""

    def __init__(self,
                 rescale,
                 colorspace,
                 current_bounds,
                 desired_bounds,
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

    def preprocess_image(self,
                         path):
        """
        Pipeline for complete preprocessing of an image.
        Loads image, rescales it, converts colorspace, normalizes, and converts datatype.
        Finalizes by converting HWC to NHWC.

        Parameters:
            - For each parameter, see the function(s) listed in the imops module for details.
                - path
                    - valid_file
                    - load_image
                - rescale
                    - resize_image
                - colorspace
                    - convert_colorspace
                - current_bounds, desired_bounds, dtype
                    - normalize_image
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
        An extension of `ImagePreprocessor.preprocess_image` for directories.
        Given a directory, preprocesses images in it with `ImagePreprocessor.preprocess_image`.
        Subdirectories and files of unsupported formats are ignored.

        Parameters:
            - path (str)
                - Path to the directory.
            - For other parameters, see the preprocess_image function in this module for details.
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
            - encoding (dict)
                - Maps the name of the subdirectory (class) to a label.
                - ex: {'cats': [1, 0], 'dogs': [0, 1]}
        Yields:
            - A tuple (step, preprocessed_image_array, label_array) starting from step 1.
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
                if not imops.valid_file(image_path):
                    continue
                preprocessed_image = self.preprocess_image(image_path)
                label = np.expand_dims(encoding[class_], axis=0).astype('float32')
                if cursors[class_] == (len(images[class_]) - 1):
                    cursors[class_] = 0
                else:
                    cursors[class_] += 1
        yield step, preprocessed_image, label
