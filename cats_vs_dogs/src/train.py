"""Provides interface for training the model."""

import os
import shutil

import tensorflow as tf

import constants as c
from model import architecture
from preprocessing.pipelines import ImageDataPipeline


def clear_tensorboard(path):
    """
    Clears the directory for tensorboard, if it exists.

    Parameters:
        - path (str)
            - Path to the directory.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)


def train(steps,
          resuming):
    """
    Builds up a graph of operations from scratch to train a model.

    Parameters:
        - steps (int)
            - Number of gradient updates to perform.
        - resuming (bool)
            - Whether to train from scratch or resume training from a saved model.
    """
    clear_tensorboard(c.TENSORBOARD_DIR)
    sess = tf.Session()
    if resuming:
        loader = tf.train.import_meta_graph(c.SAVEMODEL_DIR + '.meta')
        loader.restore(sess, c.SAVEMODEL_DIR)
        input_ = sess.graph.get_tensor_by_name('input:0')
        label = sess.graph.get_tensor_by_name('label:0')
        optimizer = sess.graph.get_operation_by_name('optimizer')
    else:
        input_ = tf.placeholder(tf.float32, shape=c.IN_SHAPE, name='input')
        output = architecture.primary(input_, name='model')
        sess.run(tf.global_variables_initializer())
        label = tf.placeholder(tf.float32, shape=list(c.ENCODING.values())[0].shape)
        objective = tf.sqrt(tf.reduce_mean(tf.squared_difference(label, output)), name='objective')
        optimizer = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(objective, name='optimizer')
        tf.summary.scalar('objective_summary', objective)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(c.TENSORBOARD_DIR, graph=sess.graph)
    preprocessor = ImageDataPipeline(c.IN_SHAPE[1:3],
                                     c.COLORSPACE,
                                     [[0, 255], [0, 255], [0, 255]],
                                     [[0, 1], [-1, 1], [-1, 1]])
    for step, input_arg, label_arg in preprocessor.preprocess_classes(steps, c.TRAIN_DIR,
                                                                      c.ENCODING):
        print('Step: {}/{}'.format(step, steps))
        _, step_summary = sess.run([optimizer, summary],
                                   feed_dict={input_: input_arg, label: label_arg})
        writer.add_summary(step_summary, global_step=step)
    tf.train.Saver().save(sess, c.SAVEMODEL_DIR)
    print('\a')
