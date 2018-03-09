"""Provides an interface for interacting with the model."""

import argparse
import os
import shutil

import numpy as np
import tensorflow as tf

import constants as c
from architecture import model
from preprocessing import ImagePreprocessor


def train(steps, resuming):
    """
    Trains the model and saves the result.

    # Parameters
        steps (int): Amount of images to train on.
        resuming (bool): Whether or not to resume training on a saved model.
    """
    if c.TENSORBOARD_DIR in os.listdir():
        shutil.rmtree(c.TENSORBOARD_DIR)
    if not resuming:
        input_ = tf.placeholder(tf.float32, shape=[c.BATCH, c.ROWS, c.COLS, c.CHAN], name='input')
        output = model(input_)
        label = tf.placeholder(tf.float32, shape=[1, 2], name='label')
        objective = tf.identity(tf.losses.absolute_difference(label, output), name='objective')
        optimizer = tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True,
                                               name='optimizer').minimize(objective)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            tf.summary.scalar('objective', objective)
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(c.TENSORBOARD_DIR, graph=tf.get_default_graph())
            for step, input_arg, label_arg in ImagePreprocessor().preprocess_directory(
                    steps, c.TRAIN_DIR, c.ENCODING, [c.COLS, c.ROWS]):
                print('Step: {}/{}'.format(step, steps))
                sess.run(optimizer, feed_dict={input_: input_arg, label: label_arg})

                step_summary = sess.run(summary, feed_dict={input_: input_arg, label: label_arg})
                writer.add_summary(step_summary, global_step=step)
            tf.train.Saver().save(sess, c.SAVEMODEL_DIR)
    else:
        with tf.Session() as sess:
            loader = tf.train.import_meta_graph(c.SAVEMODEL_DIR + '.meta')
            loader.restore(sess, c.SAVEMODEL_DIR)
            graph = tf.get_default_graph()
            input_ = graph.get_tensor_by_name('input:0')
            label = graph.get_tensor_by_name('label:0')
            objective = graph.get_tensor_by_name('objective:0')
            optimizer = graph.get_operation_by_name('optimizer')

            tf.summary.scalar('objective', objective)
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(c.TENSORBOARD_DIR, graph=tf.get_default_graph())
            for step, input_arg, label_arg in ImagePreprocessor().preprocess_directory(
                    steps, c.TRAIN_DIR, c.ENCODING, [c.COLS, c.ROWS]):
                print('Step: {}/{}'.format(step, steps))
                sess.run(optimizer, feed_dict={input_: input_arg, label: label_arg})

                step_summary = sess.run(summary, feed_dict={input_: input_arg, label: label_arg})
                writer.add_summary(step_summary, global_step=step)
            tf.train.Saver().save(sess, c.SAVEMODEL_DIR)


def classify(image):
    """
    Classify a single image.

    # Parameters
        image (str): Path to the image in question.
    # Prints
        The resulting tensor of predictions.
        In this case, argmax==0 means 'cat' and argmax==1 means 'dog'.
    """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(c.SAVEMODEL_DIR + '.meta')
        loader.restore(sess, c.SAVEMODEL_DIR)
        graph = tf.get_default_graph()
        input_ = graph.get_tensor_by_name('input:0')
        output = graph.get_tensor_by_name('output:0')
        input_arg = np.array([ImagePreprocessor().preprocess_image(image, [256, 256])])
        result = sess.run(output, feed_dict={input_: input_arg})
        if np.argmax(result) == 0:
            print('cat')
        else:
            print('dog')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resuming', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--image')
    parser.set_defaults(resuming=False)
    args = parser.parse_args()
    if args.train:
        train(args.steps, args.resuming)
    elif args.classify:
        classify(args.image)
