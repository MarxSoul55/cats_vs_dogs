"""Provides an interface for interacting with the model."""

import argparse
import msvcrt
import os
import shutil
import sys
import numpy as np
import tensorflow as tf

import architecture
import constants as c
from preprocessing import ImagePreprocessor


def train(steps, resuming):
    """
    Trains the model and saves the result.

    # Parameters
        steps (int):
            - Amount of images to train on.
        resuming (bool):
            - Whether to train from scratch or resume training from a saved model.
    """
    if c.TENSORBOARD_DIR in os.listdir():
        shutil.rmtree(c.TENSORBOARD_DIR)
    with tf.Session() as sess:
        if resuming:
            saver = tf.train.import_meta_graph(c.SAVEMODEL_DIR + '.meta')
            saver.restore(sess, c.SAVEMODEL_DIR)
            graph = tf.get_default_graph()
            input_ = graph.get_tensor_by_name('input:0')
            label = graph.get_tensor_by_name('label:0')
            optimizer = graph.get_operation_by_name('optimizer')
        else:
            input_ = tf.placeholder(tf.float32, shape=[1, c.ROWS, c.COLS, c.CHAN], name='input')
            model = architecture.model(input_, name='model')
            label = tf.placeholder(tf.float32, shape=c.LABEL_SHAPE, name='label')
            objective = tf.sqrt(tf.reduce_mean(tf.squared_difference(label, model)),
                                name='objective')
            optimizer = tf.train.MomentumOptimizer(0.0001, 0.9,
                                                   use_nesterov=True).minimize(objective,
                                                                               name='optimizer')
            tf.summary.scalar('objective_summary', objective)
            sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(c.TENSORBOARD_DIR, graph=sess.graph)
        preprocessor = ImagePreprocessor([c.COLS, c.ROWS], c.COLOR_SPACE)
        for step, input_arg, label_arg in preprocessor.preprocess_classes(steps, c.TRAIN_DIR,
                                                                          c.ENCODING):
            print('Step: {}/{}'.format(step, steps))
            _, step_summary = sess.run([optimizer, summary],
                                       feed_dict={input_: input_arg, label: label_arg})
            writer.add_summary(step_summary, global_step=step)
        tf.train.Saver().save(sess, c.SAVEMODEL_DIR)
        print('\a')


def classify(path):
    """
    Does one of 3 things:
    1. Given a path to an image file on disk (WITH A FILE-EXTENSION), classifies it.
    2. Given a path to a directory on disk, classifies all images found in it (excluding
       subdirectories and files with unsupported formats).
    3. Given a URL to an image, classifies it.

    # Parameters
        path (str):
            - Can be a normal path to an image on disk.
            - Can also be a URL that returns an image.
    # Returns
        - If given path to an image file on disk, or a URL to an image, returns a string that is
          either 'cat' or 'dog'.
        - If given path to a directory, returns a dictionary {'filename': 'guessed animal'}
    """
    preprocessor = ImagePreprocessor([c.COLS, c.ROWS], c.COLOR_SPACE)
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(c.SAVEMODEL_DIR + '.meta')
        loader.restore(sess, c.SAVEMODEL_DIR)
        graph = tf.get_default_graph()
        input_ = graph.get_tensor_by_name('input:0')
        model_output = graph.get_tensor_by_name('model/output/output:0')
        if os.path.isdir(path):
            results = {}
            for image_name, preprocessed_image in preprocessor.preprocess_directory(path):
                input_arg = np.expand_dims(preprocessed_image, axis=0)
                result = sess.run(model_output, feed_dict={input_: input_arg})
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Need help? See: https://github.com/MarxSoul55/cats_vs_dogs#using-the-cli')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resuming', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--source')
    parser.set_defaults(resuming=False)
    args = parser.parse_args()
    if args.train:
        print('WARNING: Training will overwrite the saved model (if it exists). EXECUTE Y/N?')
        if msvcrt.getch().decode().lower() == 'y':
            train(args.steps, args.resuming)
        else:
            sys.exit('Program closed.')
    elif args.classify:
        print(classify(args.source))
