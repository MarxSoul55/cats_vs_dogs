"""Provides an interface for interacting with the model."""

import argparse

import numpy as np
import tensorflow as tf

from test_architecture import model
from test_preprocessing import ImagePreprocessor


def train(steps, resuming):
    """
    Trains the model and saves the result.

    # Parameters
        steps (int): Amount of images to train on.
        resuming (bool): Whether or not to resume training on a saved model.
    """
    encoding = {'cats': [1, 0], 'dogs': [0, 1]}
    preprocessor = ImagePreprocessor()
    if not resuming:
        input_ = tf.placeholder(tf.float32, shape=[1, 256, 256, 3], name='input')
        labels = tf.placeholder(tf.float32, shape=[1, 2], name='labels')
        output = model(input_)
        objective = tf.losses.absolute_difference(labels, output,
                                                  reduction=tf.losses.Reduction.MEAN)
        optimizer = tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True,
                                               name='optimizer').minimize(objective)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step, input_arg, label_arg in preprocessor.preprocess_directory(steps,
                                                                                'data/train',
                                                                                encoding,
                                                                                [256, 256]):
                print('Step: {}/{}'.format(step, steps))
                sess.run(optimizer, feed_dict={input_: input_arg, labels: label_arg})
            tf.train.Saver().save(sess, 'saved/model')
    else:
        with tf.Session() as sess:
            loader = tf.train.import_meta_graph('saved/model.meta')
            loader.restore(sess, 'saved/model')
            graph = tf.get_default_graph()
            input_ = graph.get_tensor_by_name('input:0')
            labels = graph.get_tensor_by_name('labels:0')
            optimizer = graph.get_operation_by_name('optimizer')
            for step, input_arg, label_arg in preprocessor.preprocess_directory(steps,
                                                                                'data/train',
                                                                                encoding,
                                                                                [256, 256]):
                print('Step: {}/{}'.format(step, steps))
                sess.run(optimizer, feed_dict={input_: input_arg, labels: label_arg})
            tf.train.Saver().save(sess, 'saved/model')


def classify(image):
    """
    Classify a single image.

    # Parameters
        image (str): Path to the image in question.
    # Prints
        The resulting tensor of predictions.
        In this case, argmax==0 means 'cat' and argmax==1 means 'dog'.
    """
    data = tf.placeholder(tf.float32, shape=[1, 256, 256, 3])
    output = model(data)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'saved/model')
        preprocessor = ImagePreprocessor()
        data_arg = np.array([preprocessor.preprocess_image(image, (256, 256))])
        result = output.eval(feed_dict={data: data_arg})
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resuming', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--path')
    parser.set_defaults(resuming=False)
    args = parser.parse_args()
    if args.train:
        train(args.steps, args.resuming)
    elif args.classify:
        classify(args.path)
