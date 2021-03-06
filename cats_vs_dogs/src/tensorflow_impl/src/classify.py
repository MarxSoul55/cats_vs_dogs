"""Provides interface to classification with the model."""

import os

import numpy as np
import tensorflow as tf

from .preprocessing import ImageDataPipeline


def predicted_label(prediction_tensor,
                    label_dict):
    """
    Generates the predicted label by comparing the tensor prediction to `label_dict`.

    Parameters:
        - prediction (np.ndarray)
            - The prediction as represented by the model's tensor output.
        - label_dict (dict, str --> np.ndarray)
            - See the parent function for details.
    Returns:
        - A string; the predicted label.
    """
    classes = list(label_dict.keys())
    labels = list(label_dict.values())
    differences = []
    for label in labels:
        l2_difference = np.sqrt(np.sum((label - prediction_tensor) ** 2))
        differences.append(l2_difference)
    index_of_smallest_difference = differences.index(min(differences))
    return classes[index_of_smallest_difference]


def main(src,
         model_savepath,
         label_dict):
    """
    Does one of 2 things:
    1. Given a path to an image file on disk (WITH A FILE-EXTENSION), classifies it.
    2. Given a path to a directory on disk, classifies all images found in it (excluding
       subdirectories and files with unsupported formats).

    Parameters:
        - src (str)
            - Can be a normal path to an image on disk.
            - Can also be a path to a directory.
        - model_savepath (str)
            - Path to a saved model on disk.
            - This model is used for the classification.
            - e.g. X/Y where Y is the prefix of the .meta, .index, etc. files for TensorFlow.
        - label_dict (dict, str --> np.ndarray)
            - Maps the name of the subdirectory (class) to a label.
                - ex: {'cats': np.array([[1, 0]]), 'dogs': np.array([[0, 1]])}
                - Each label must have the same shape!
            - The model's prediction is compared to each label.
                - Whichever label differs the least from the prediction is chosen to return.
    Returns:
        - If given path to an image file on disk, returns a string that is either 'cat' or 'dog'.
        - If given path to a directory, returns a dictionary {'filename': 'guessed animal'}
    """
    # Load the model and placeholder input.
    sess = tf.Session()
    loader = tf.train.import_meta_graph(model_savepath + '.meta')
    loader.restore(sess, model_savepath)
    input_ = sess.graph.get_tensor_by_name('input:0')
    model = sess.graph.get_tensor_by_name('model/output/output:0')
    preprocessor = ImageDataPipeline(mode='NHWC')
    # If path leads to a directory, classify each image in it.
    if os.path.isdir(src):
        results = {}
        for image_name, preprocessed_image in preprocessor.preprocess_directory(src):
            prediction_tensor = sess.run(model, feed_dict={input_: preprocessed_image})
            results[image_name] = predicted_label(prediction_tensor, label_dict)
        return results
    # Else, classify one image.
    preprocessed_image = preprocessor.preprocess_image(src)
    prediction_tensor = sess.run(model, feed_dict={input_: preprocessed_image})
    return predicted_label(prediction_tensor, label_dict)
