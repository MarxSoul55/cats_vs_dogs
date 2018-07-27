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


def build_graph(sess,
                input_shape,
                label_shape):
    """
    Builds a graph for the training process.

    Parameters:
        - sess (tf.Session)
            - A session object where the graph will be loaded.
        - input_shape (list)
            - Shape of the input tensor.
        - label_shape (list)
            - Shape of the label tensor.
    Returns:
        - The variables necessary for the training loop.
            - An input placeholder.
            - A label placeholder.
            - An optimizer.minimize method from TensorFlow.
    """
    input_ = tf.placeholder(tf.float32, shape=input_shape, name='input')
    output = architecture.primary(input_, name='model')
    sess.run(tf.global_variables_initializer())
    label = tf.placeholder(tf.float32, shape=label_shape)
    objective = tf.sqrt(tf.reduce_mean(tf.squared_difference(label, output)), name='objective')
    optimizer = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(objective, name='optimizer')
    tf.summary.scalar('objective_summary', objective)
    return input_, label, optimizer


def load_graph(sess,
               savemodel_dir):
    """
    Loads an existing graph from a directory saved to by a tf.train.Saver object.

    Parameters:
        - savemodel_dir (str)
            - X/Y
                - X is the name of the directory that holds saved data about the model.
                - Y is the prefix for the .data, .index, and .meta files.
    Returns:
        - The variables necessary for the training loop.
            - An input placeholder.
            - A label placeholder.
            - An optimizer.minimize method from TensorFlow.
    """
    saver = tf.train.import_meta_graph(savemodel_dir + '.meta')
    saver.restore(sess, savemodel_dir)
    input_ = sess.graph.get_tensor_by_name('input:0')
    label = sess.graph.get_tensor_by_name('label:0')
    optimizer = sess.graph.get_operation_by_name('optimizer')
    return input_, label, optimizer


def initialize_summary_nodes(tensorboard_dir):
    """
    Merges all summary ops on graph to a single op, and initializes a FileWriter object.

    Parameters:
        - tensorboard_dir (str)
            - Path to the directory where the FileWriter object will store summaries.
    Returns:
        - summary (a tf operation)
            - A node on the graph which collects the specified data.
        - writer (tf.summary.FileWriter)
            - Object which saves summaries to a directory.
    """
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, graph=self.sess.graph)
    return summary, writer


def main(sess,
         optimizer,
         steps,
         input_,
         label,
         summary,
         writer,
         savedir,
         train_dir,
         encoding,
         rescale,
         colorspace,
         current_bounds,
         desired_bounds,
         dtype='float32'):
    """
    Implements the training loop and saves the resulting model.

    Parameters:
        - sess (tf.Session)
            - Session that the graph will run on.
        - steps (int)
            - Number of gradient updates to perform (i.e. number of images to train on)
        - input_, label
            - See train.build_graph or train.load_graph for details.
        - summary, writer
            - See train.initialize_summary_nodes for details.
        - savedir (str)
            - Directory to save the model to.
        - train_dir, encoding, rescale, colorspace, current_bounds, desired_bounds, dtype
            - See preprocessing.pipelines.ImageDataPipeline for details.
    """
    preprocessor = ImageDataPipeline(rescale, colorspace, current_bounds, desired_bounds,
                                     dtype=dtype)
    for step, input_path, input_arg, label_arg in preprocessor.preprocess_classes(steps, train_dir,
                                                                                  encoding):
        _, step_summary = sess.run([optimizer, summary],
                                   feed_dict={input_: input_arg, label: label_arg})
        writer.add_summary(step_summary, global_step=step)
    tf.train.Saver().save(sess, savedir)
    print('\a')
