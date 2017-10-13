"""Provides an interface for interacting with the model."""

import argparse

import numpy as np
import tensorflow as tf

from layers.accuracies import categorical_accuracy_reporter
from layers.activations import elu, sigmoid
from layers.convolutional import convolution_2d, flatten_2d, globalaveragepooling_2d, maxpooling_2d
from layers.core import dense
from layers.objectives import mean_binary_entropy
from layers.optimizers import nesterov_momentum
from layers.preprocessing import ImagePreprocessor
from layers.serving import predict_binary, restore_protobuf, save_protobuf
from layers.training import restore_model, save_model

DATA_DIR = 'data/train'


def model(input_):
    """
    Defines the model's architecture.

    # Parameters
        input_ (tf.placeholder): Placeholder for the input data.
    # Returns
        The output of the model.
    """
    output = convolution_2d(input_, 8)
    output = elu(output)
    output = maxpooling_2d(output)
    output = convolution_2d(output, 16)
    output = elu(output)
    output = convolution_2d(output, 16)
    output = elu(output)
    output = maxpooling_2d(output)
    output = convolution_2d(output, 32)
    output = elu(output)
    output = convolution_2d(output, 32)
    output = elu(output)
    output = convolution_2d(output, 32)
    output = elu(output)
    output = maxpooling_2d(output)
    output = convolution_2d(output, 64)
    output = elu(output)
    output = convolution_2d(output, 64)
    output = elu(output)
    output = convolution_2d(output, 64)
    output = elu(output)
    output = convolution_2d(output, 64)
    output = elu(output)
    output = maxpooling_2d(output)
    output = globalaveragepooling_2d(output)
    output = flatten_2d(output)
    output = dense(output, 2)
    return output


def train(steps, resuming, save_pb):
    """
    Trains the model and saves the result.

    # Parameters
        steps (int): Amount of images to train on.
        resuming (bool): Whether or not to train from scratch.
        save_pb (bool): Whether or not to save a protobuf.
    """
    with tf.name_scope('input'):
        data = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        labels = tf.placeholder(tf.float32, shape=[None, 2])
    with tf.name_scope('output'):
        logits = model(data)
        output = sigmoid(logits)
    with tf.name_scope('objective'):
        objective = mean_binary_entropy(labels, logits)
    with tf.name_scope('accuracy'):
        accuracy = categorical_accuracy_reporter(labels, output)
    with tf.name_scope('optimizer'):
        optimizer = nesterov_momentum(objective)
    tf.summary.scalar('objective', objective)
    tf.summary.scalar('accuracy', accuracy)
    summary = tf.summary.merge_all()
    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter('tensorboard', graph=tf.get_default_graph())
        if resuming:
            restore_model(sess)
        prepro = ImagePreprocessor()
        for step, data_arg, label_arg in prepro.preprocess_directory(steps, 'data/train',
                                                                     ['cats', 'dogs'],
                                                                     rescale=(256, 256)):
            print('Step: {}/{}'.format(step, steps))
            optimizer.run(feed_dict={data: data_arg, labels: label_arg})
            current_summary = summary.eval(feed_dict={data: data_arg, labels: label_arg})
            writer.add_summary(current_summary, step)
        save_model(sess)
        if save_pb:
            save_protobuf(sess, 'cats_vs_dogs')


def serve(image):
    """
    Serve the model on a single image.

    # Parameters
        image (str): Path to the image in question.
    # Returns
        A string; either 'cat' or 'dog'.
    """
    pass  # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', action='store_true')
    parser.add_argument('-r', '--resuming', action='store_true')
    parser.add_argument('-s', '--steps', type=int)
    parser.add_argument('-sp', '--saveprotobuf', action='store_true')
    parser.add_argument('-se', '--serve', action='store_true')
    parser.add_argument('-i', '--image')
    parser.set_defaults(resuming=False, saveprotobuf=False)
    args = parser.parse_args()
    if args.train:
        train(args.steps, args.resuming, args.saveprotobuf)
    elif args.serve:
        serve(args.image)
