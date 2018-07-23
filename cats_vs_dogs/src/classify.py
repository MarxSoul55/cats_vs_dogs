"""Provides interface to classification with the model."""

import os

import numpy as np
import tensorflow as tf

import constants as c
from model.pipelines import ImageDataPipeline


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
        - encoding (dict, str --> np.ndarray)
            - Maps the name of the subdirectory (class) to a label.
                - ex: {'cats': np.array([[1, 0]]), 'dogs': np.array([[0, 1]])}
                - Each label must have the same shape!
            - The model's prediction is compared to each label.
                - Whichever label differs the least from the prediction is chosen to return.
    Returns:
        - If given path to an image file on disk, returns a string that is either 'cat' or 'dog'.
        - If given path to a directory, returns a dictionary {'filename': 'guessed animal'}
    """
    def predicted_label(prediction_tensor,
                        encoding):
        """
        Generates the predicted label by comparing the tensor prediction to `encoding`.

        Parameters:
            - prediction (np.ndarray)
                - The prediction as represented by the model's tensor output.
            - encoding (dict, str --> np.ndarray)
                - See the parent function for details.
        Returns:
            - A string; the predicted label.
        """
        classes = list(encoding.keys())
        labels = list(encoding.values())
        differences = []
        for label in labels:
            l2_difference = np.sqrt(np.sum((label - prediction_tensor) ** 2))
            differences.append(l2_difference)
        index_of_smallest_difference = differences.index(min(differences))
        return classes[index_of_smallest_difference]
    sess = tf.Session()
    loader = tf.train.import_meta_graph(c.SAVEMODEL_DIR + '.meta')
    loader.restore(sess, c.SAVEMODEL_DIR)
    input_ = sess.graph.get_tensor_by_name('input:0')
    model = sess.graph.get_tensor_by_name('model/output/output:0')
    preprocessor = ImageDataPipeline(c.IN_SHAPE[1:3],
                                     c.COLORSPACE,
                                     [[0, 255], [0, 255], [0, 255]],
                                     [[0, 1], [-1, 1], [-1, 1]])
    if os.path.isdir(path):
        results = {}
        for image_name, preprocessed_image in preprocessor.preprocess_directory(path):
            prediction_tensor = sess.run(model, feed_dict={input_: preprocessed_image})
            results[image_name] = predicted_label(prediction_tensor, encoding)
        return results
    preprocessed_image = preprocessor.preprocess_image(path)
    prediction_tensor = sess.run(model, feed_dict={input_: preprocessed_image})
    return predicted_label(prediction_tensor, encoding)
