"""Provides an interface for interacting with the model."""

import argparse

import numpy as np
import tensorflow as tf

from layers.accuracies import categorical_accuracy_reporter
from layers.activations import elu
from layers.convolutional import convolution_2d, flatten_2d, globalaveragepooling_2d, maxpooling_2d
from layers.core import dense, residual
from layers.objectives import mean_absolute_error
from layers.optimizers import nesterov_momentum
from layers.preprocessing import ImagePreprocessor
from layers.training import restore_model, save_model, tensorboard_writer


def model(input_):
    """
    Defines the model's architecture.

    # Parameters
        input_ (tf.placeholder): Placeholder for the input data.
    # Returns
        The output of the model.
    """
    '''
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
    output = convolution_2d(output, 128)
    output = elu(output)
    output = convolution_2d(output, 128)
    output = elu(output)
    output = convolution_2d(output, 128)
    output = elu(output)
    output = convolution_2d(output, 128)
    output = elu(output)
    output = convolution_2d(output, 128)
    output = elu(output)
    output = maxpooling_2d(output)
    output = globalaveragepooling_2d(output)
    output = flatten_2d(output)
    output = dense(output, 2)
    return output
    '''
    # Blocks are separated by layers of pooling.
    # ---------- Block 1 ----------
    conv1 = convolution_2d(input_, 8)  # conv
    elu1 = elu(conv1)  # elu
    conv2 = convolution_2d(elu1, 8)  # conv
    residual1 = residual(input_, conv2)  # residual
    elu2 = elu(residual1)  # after-elu
    pool1 = maxpooling_2d(elu2)  # pool
    # ---------- Block 2 ----------
    conv3 = convolution_2d(pool1, 16)  # conv
    elu3 = elu(conv3)  # elu
    conv4 = convolution_2d(elu3, 16)  # conv
    residual2 = residual(pool1, conv4)  # residual
    elu4 = elu(residual2)  # after-elu
    pool2 = maxpooling_2d(elu4)  # pool
    # ---------- Block 3 ----------
    conv5 = convolution_2d(pool2, 32)  # conv
    elu5 = elu(conv5)  # elu
    conv6 = convolution_2d(elu5, 32)  # conv
    residual3 = residual(pool2, conv6)  # residual
    elu6 = elu(residual3)  # after-elu
    conv7 = convolution_2d(elu6, 32)  # conv
    elu7 = elu(conv7)  # elu
    conv8 = convolution_2d(elu7, 32)  # conv
    residual4 = residual(elu6, conv8)  # residual
    elu8 = elu(residual4)  # after-elu
    pool3 = maxpooling_2d(elu8)  # pool
    # ---------- Block 4 ----------
    conv9 = convolution_2d(pool3, 64)  # conv
    elu9 = elu(conv9)  # elu
    conv10 = convolution_2d(elu9, 64)  # conv
    residual5 = residual(pool3, conv10)  # residual
    elu10 = elu(residual5)  # after-elu
    conv11 = convolution_2d(elu10, 64)  # conv
    elu11 = elu(conv11)  # elu
    conv12 = convolution_2d(elu11, 64)  # conv
    residual6 = residual(elu10, conv12)  # residual
    elu12 = elu(residual6)  # after-elu
    pool4 = maxpooling_2d(elu12)  # pool
    # ---------- Block 5 ----------
    conv13 = convolution_2d(pool4, 128)  # conv
    elu13 = elu(conv13)  # elu
    conv14 = convolution_2d(elu13, 128)  # conv
    residual7 = residual(pool4, conv14)  # residual
    elu14 = elu(residual7)  # after-elu
    conv15 = convolution_2d(elu14, 128)  # conv
    elu15 = elu(conv15)  # elu
    conv16 = convolution_2d(elu15, 128)  # conv
    residual8 = residual(elu14, conv16)  # residual
    elu16 = elu(residual8)  # after-elu
    conv17 = convolution_2d(elu16, 128)  # conv
    elu17 = elu(conv17)  # elu
    conv18 = convolution_2d(elu17, 128)  # conv
    residual9 = residual(elu16, conv18)  # residual
    elu18 = elu(residual9)  # after-elu
    conv19 = convolution_2d(elu18, 128)  # conv
    elu19 = elu(conv19)  # elu
    conv20 = convolution_2d(elu19, 128)  # conv
    residual10 = residual(elu18, conv20)  # residual
    elu20 = elu(residual10)  # after-elu
    pool5 = maxpooling_2d(elu20)  # pool
    # ---------- Block 6 ----------
    pool6 = globalaveragepooling_2d(pool5)
    flatten1 = flatten_2d(pool6)
    output = dense(flatten1, 2)
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
    with tf.name_scope('optimizer'):
        optimizer = nesterov_momentum(objective)
    tf.summary.scalar('objective', objective)
    tf.summary.scalar('accuracy', accuracy)
    summary = tf.summary.merge_all()
    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()
        writer = tensorboard_writer()
        if resuming:
            restore_model(sess)
        preprocessor = ImagePreprocessor()
        encoding = {'cats': [1, 0], 'dogs': [0, 1]}
        for step, data_arg, label_arg in preprocessor.preprocess_directory(steps, 'data/train',
                                                                           encoding, (256, 256)):
            print('Step: {}/{}'.format(step, steps))
            optimizer.run(feed_dict={data: data_arg, labels: label_arg})
            current_summary = summary.eval(feed_dict={data: data_arg, labels: label_arg})
            writer.add_summary(current_summary, global_step=step)
        save_model(sess)


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
    with tf.name_scope('output'):
        output = model(data)
    with sess.as_default():
        # tf.global_variables_initializer().run()
        restore_model(sess)
        preprocessor = ImagePreprocessor()
        data_arg = np.array([preprocessor.preprocess_image(image, (256, 256))])
        result = output.eval(feed_dict={data: data_arg})
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
