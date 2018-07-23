"""Provides interface to classification with the model."""

import os

import numpy as np
import tensorflow as tf

import constants as c
from model.pipelines import ImageDataPipeline


def l2_error(a, b):
    """
    Calculates the l2 error between two arrays.

    Parameters:
        - a (np.ndarray)
        - b (np.ndarray)
            - Must have same shape as `a`.
    Returns:
        - A float representing the error.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def classify(path,
             encoding):
    """
    Does one of 2 things:
    1. Given a path to an image file on disk (WITH A FILE-EXTENSION), classifies it.
    2. Given a path to a directory on disk, classifies all images found in it (excluding
       subdirectories and files with unsupported formats).

    Parameters:
        - path (str)
            - Can be a normal path to an image on disk.
            - Can also be a path to a directory.
    Returns:
        - If given path to an image file on disk, returns a string that is either 'cat' or 'dog'.
        - If given path to a directory, returns a dictionary {'filename': 'guessed animal'}
    """
    preprocessor = ImageDataPipeline(c.IN_SHAPE[1:3], c.COLORSPACE, [[0, 255], [0, 255], [0, 255]],
                                     [[0, 1], [-1, 1], [-1, 1]])
    sess = tf.Session()
    loader = tf.train.import_meta_graph(c.SAVEMODEL_DIR + '.meta')
    loader.restore(sess, c.SAVEMODEL_DIR)
    input_ = sess.graph.get_tensor_by_name('input:0')
    model_output = sess.graph.get_tensor_by_name('model/output/output:0')
    if os.path.isdir(path):
        results = {}
        for image_name, preprocessed_image in preprocessor.preprocess_directory(path):
            result = sess.run(model_output, feed_dict={input_: preprocessed_image})
            if np.argmax(result) == 0:
                results[image_name] = 'cat'
            else:
                results[image_name] = 'dog'
        return results
    input_arg = np.expand_dims(preprocessor.preprocess_image(path), axis=0)
    result = sess.run(model_output, feed_dict={input_: input_arg})
    if np.argmax(result) == np.argmax(c.ENCODING['cats']):
        return 'cat'
    return 'dog'
