"""Pipelines which perform complete image preprocessing."""

import os
import random

import numpy as np

import imops


def preprocess_image(path,
                     rescale,
                     colorspace,
                     current_bounds,
                     desired_bounds,
                     dtype='float32'):
    """
    Pipeline for complete preprocessing of an image.
    Loads image, rescales it, converts colorspace, normalizes, and converts datatype.

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
        - The fully preprocessed image.
    """
    image = imops.load_image(path)
    image = imops.resize_image(image, rescale)
    image = imops.convert_colorspace(image, colorspace)
    image = imops.normalize_image(image, current_bounds, desired_bounds, dtype=dtype)
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_directory(path,
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
        if not imops.valid_file(filepath):
            continue
        preprocessed_image = preprocess_image(filepath, rescale, colorspace,
                                              current_bounds, desired_bounds, dtype=dtype)
        yield filename, preprocessed_image


def preprocess_classes(steps,
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
