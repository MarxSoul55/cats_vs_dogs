"""Provides interface for training the model."""

import os
import shutil

import tensorflow as tf

import constants as c


def clear_tensorboard(path):
    """
    Clears the directory for tensorboard, if it exists.

    Parameters:
        - path (str)
            - Path to the directory.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)


def train(steps, resuming):
    """
    Trains the model and saves the result.

    # Parameters
        steps (int):
            - Amount of images to train on.
        resuming (bool):
            - Whether to train from scratch or resume training from a saved model.
    """
    clear_tensorboard(c.TENSORBOARD_DIR)
    sess = tf.Session()
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
        objective = tf.sqrt(tf.reduce_mean(tf.squared_difference(label, model)), name='objective')
        optimizer = tf.train.MomentumOptimizer(
            0.0001, 0.9, use_nesterov=True).minimize(objective, name='optimizer')
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
    sess = tf.Session()
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
        objective = tf.sqrt(tf.reduce_mean(tf.squared_difference(label, model)), name='objective')
        optimizer = tf.train.MomentumOptimizer(
            0.0001, 0.9, use_nesterov=True).minimize(objective, name='optimizer')
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
