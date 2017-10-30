"""Provides an interface for interacting with the model."""

import argparse
import os
import shutil

import numpy as np
import tensorflow as tf

from preprocessing import ImagePreprocessor


def model(input_):
    """
    Defines the model's architecture.

    # Parameters
        input_ (tf.placeholder): Placeholder for the input data.
    # Returns
        The output of the model.
    """
    output = tf.layers.conv2d(input_, 8, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output, 16, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 16, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output, 32, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 32, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 32, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output, 64, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 64, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 64, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 64, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.average_pooling2d(output, output.shape.as_list()[1], 1)
    output = tf.reshape(output, [1, output.shape.as_list()[3]])
    output = tf.layers.dense(output, 2, kernel_initializer=tf.orthogonal_initializer())
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
        objective = tf.losses.absolute_difference(labels, output,
                                                  reduction=tf.losses.Reduction.MEAN)
    with tf.name_scope('accuracy'):
        bools = tf.equal(tf.argmax(labels, axis=1), tf.argmax(output, axis=1))
        accuracy = tf.reduce_mean(tf.cast(bools, tf.float32))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True).minimize(objective)
    tf.summary.scalar('objective', objective)
    tf.summary.scalar('accuracy', accuracy)
    summary = tf.summary.merge_all()
    sess = tf.Session()
    with sess:
        saver = tf.train.Saver()
        if resuming:
            saver.restore(sess, 'saved_model/saved_model')
        else:
            tf.global_variables_initializer().run()
        if 'tensorboard' in os.listdir():
            shutil.rmtree('tensorboard')
        writer = tf.summary.FileWriter('tensorboard', graph=tf.get_default_graph())
        preprocessor = ImagePreprocessor()
        encoding = {'cats': [1, 0], 'dogs': [0, 1]}
        for step, data_arg, label_arg in preprocessor.preprocess_directory(steps, 'data/train',
                                                                           encoding, (256, 256)):
            print('Step: {}/{}'.format(step, steps))
            optimizer.run(feed_dict={data: data_arg, labels: label_arg})
            current_summary = summary.eval(feed_dict={data: data_arg, labels: label_arg})
            writer.add_summary(current_summary, global_step=step)
        saver.save(sess, 'saved_model/saved_model')


def test(image):
    """
    Test the model on a single image.

    # Parameters
        image (str): Path to the image in question.
    # Prints
        The resulting tensor of predictions.
        In this case, argmax==0 means 'cat' and argmax==1 means 'dog'.
    """
    data = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    with tf.name_scope('output'):
        output = model(data)
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, 'saved_model/saved_model')
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
