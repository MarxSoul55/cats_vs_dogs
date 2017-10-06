"""Provides an interface for interacting with the model."""

import argparse
from collections import deque

import tensorflow as tf

from layers.activations import elu, sigmoid
from layers.convolutional import convolution_2d, flatten_2d, globalaveragepooling_2d, maxpooling_2d
from layers.core import dense
from layers.meta import restore_model, save_model
from layers.objectives import mean_binary_entropy
from layers.optimizers import nesterov_momentum
from layers.preprocessing import ImagePreprocessor
from layers.reporters import accuracy_reporter, report

# Inside of this directory, there should be 2 more directories, `cats` and `dogs`.
# Those directories will contain the actual images.
DATA_DIR = 'data/train'


def model(input_):
    """
    Defines the model's architecture.

    # Parameters
        input_ (tf.placeholder): Placeholder for the input data.
    # Returns
        The output of the model.
    """
    output = convolution_2d(input_, 32)
    output = elu(output)
    output = maxpooling_2d(output)
    output = convolution_2d(output, 64)
    output = elu(output)
    output = maxpooling_2d(output)
    output = convolution_2d(output, 128)
    output = elu(output)
    output = maxpooling_2d(output)
    output = convolution_2d(output, 256)
    output = elu(output)
    output = maxpooling_2d(output)
    output = convolution_2d(output, 512)
    output = elu(output)
    output = maxpooling_2d(output)
    output = globalaveragepooling_2d(output)
    output = flatten_2d(output)
    output = dense(output, 2)
    return output


def train(steps, resuming):
    """
    Trains the model with SGD + Momentum.

    # Parameters
        steps (int): Amount of images to train on.
        resuming (bool): Whether or not to train from scratch.
    """
    # Create placeholders and define operations.
    data = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
    labels = tf.placeholder(tf.float32, shape=[None, 2])
    logits = model(data)
    output = sigmoid(logits)
    objective = mean_binary_entropy(labels, logits)
    accuracy = accuracy_reporter(labels, output)
    optimizer = nesterov_momentum(objective)
    # Create session, initialize global-variables and saver.
    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()
        if resuming:
            restore_model(sess)
        # Create preprocessor and `order` argument.
        prepro = ImagePreprocessor()
        order = ['cats', 'dogs']
        # Deque of moving-average of accuracies for reporting-purposes.
        accuracies = deque()
        for step, data_arg, label_arg in prepro.preprocess_directory(steps, 'data/train', order,
                                                                     rescale=(128, 128)):
            current_accuracy = accuracy.eval(feed_dict={data: data_arg, labels: label_arg})
            accuracies.append(current_accuracy)
            if len(accuracies) == 100:
                moving_accuracy = sum(accuracies) / len(accuracies)
                accuracies.popleft()
            else:
                moving_accuracy = 'WTNG'
            current_objective = objective.eval(feed_dict={data: data_arg, labels: label_arg})
            report(step, steps, moving_accuracy, current_objective)
            optimizer.run(feed_dict={data: data_arg, labels: label_arg})
        save_model(sess)


def test():
    # TODO
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', action='store_true')
    parser.add_argument('-r', '--resuming', action='store_true')
    parser.add_argument('-s', '--steps', type=int)
    parser.add_argument('-te', '--test', action='store_true')
    parser.set_defaults(resuming=False)
    args = parser.parse_args()
    if args.train:
        train(args.steps, args.resuming)
    elif args.test:
        test()
