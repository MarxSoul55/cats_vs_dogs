"""Provides an interface for interacting with the model."""

import argparse

import numpy as np
import tensorflow as tf

from layers.core.accuracies import categorical_accuracy_reporter
from layers.core.activations import elu
from layers.core.convolutional import conv_2d, flatten_2d, global_avg_pool_2d, max_pool_2d
from layers.core.misc import dense
from layers.core.objectives import mean_absolute_error
from layers.core.optimizers import nesterov_momentum
from layers.core.preprocessing import ImagePreprocessor
from layers.core.training import tensorboard_writer
from layers.core.serving import restore_protobuf, save_protobuf


def model(input_):
    """
    Defines the model's architecture.

    # Parameters
        input_ (tf.placeholder): Placeholder for the input data.
    # Returns
        The output of the model.
    """
    output = conv_2d(input_, 8)
    output = elu(output)
    output = max_pool_2d(output)
    output = conv_2d(output, 16)
    output = elu(output)
    output = conv_2d(output, 16)
    output = elu(output)
    output = max_pool_2d(output)
    output = conv_2d(output, 32)
    output = elu(output)
    output = conv_2d(output, 32)
    output = elu(output)
    output = conv_2d(output, 32)
    output = elu(output)
    output = max_pool_2d(output)
    output = conv_2d(output, 64)
    output = elu(output)
    output = conv_2d(output, 64)
    output = elu(output)
    output = conv_2d(output, 64)
    output = elu(output)
    output = conv_2d(output, 64)
    output = elu(output)
    output = max_pool_2d(output)
    output = conv_2d(output, 128)
    output = elu(output)
    output = conv_2d(output, 128)
    output = elu(output)
    output = conv_2d(output, 128)
    output = elu(output)
    output = conv_2d(output, 128)
    output = elu(output)
    output = conv_2d(output, 128)
    output = elu(output)
    output = max_pool_2d(output)
    output = global_avg_pool_2d(output)
    output = flatten_2d(output)
    output = dense(output, 2)
    # return tf.Variable(output, name='model', expected_shape=[1, 2])
    return output


def train(steps, resuming):
    """
    Trains the model and saves the result.

    # Parameters
        steps (int): Amount of images to train on.
        resuming (bool): Whether or not to resume training on a saved model.
    """
    with tf.name_scope('input'):
        data = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        labels = tf.placeholder(tf.float32, shape=[None, 2])
    with tf.name_scope('output'):
        output = model(data)
    with tf.name_scope('objective'):
        objective = mean_absolute_error(labels, output)
    with tf.name_scope('accuracy'):
        accuracy = categorical_accuracy_reporter(labels, output)
    # with tf.name_scope('optimizer'):
    optimizer = nesterov_momentum(objective)
    tf.summary.scalar('objective', objective)
    tf.summary.scalar('accuracy', accuracy)
    summary = tf.summary.merge_all()
    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()
        writer = tensorboard_writer()
        if resuming:
            restore_protobuf(sess, 'cats_vs_dogs')
            tf.reset_default_graph()
        preprocessor = ImagePreprocessor()
        encoding = {'cats': [1, 0], 'dogs': [0, 1]}
        for step, data_arg, label_arg in preprocessor.preprocess_directory(steps, 'data/train',
                                                                           encoding, (256, 256)):
            print('Step: {}/{}'.format(step, steps))
            optimizer.run(feed_dict={data: data_arg, labels: label_arg})
            current_summary = summary.eval(feed_dict={data: data_arg, labels: label_arg})
            writer.add_summary(current_summary, global_step=step)
        save_protobuf(sess, 'cats_vs_dogs')


def test(image):
    """
    Test the model on a single image.

    # Parameters
        image (str): Path to the image in question.
    # Prints
        The resulting tensor of predictions.
        In this case, argmax==0 means 'cat' and argmax==1 means 'dog'.
    """
    sess = tf.Session()
    data = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    # with tf.name_scope('output'):
    output = model(data)
    with sess.as_default():
        restore_protobuf(sess, 'cats_vs_dogs')
        tf.global_variables_initializer().run()
        preprocessor = ImagePreprocessor()
        data_arg = np.array([preprocessor.preprocess_image(image, (256, 256))])
        # result = output.eval(feed_dict={data: data_arg})
        result = sess.run(output, feed_dict={data: data_arg})
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', action='store_true')
    parser.add_argument('-r', '--resuming', action='store_true')
    parser.add_argument('-s', '--steps', type=int)
    parser.add_argument('-te', '--test', action='store_true')
    parser.add_argument('-i', '--image')
    parser.set_defaults(resuming=False)
    args = parser.parse_args()
    if args.train:
        train(args.steps, args.resuming)
    elif args.test:
        test(args.image)
