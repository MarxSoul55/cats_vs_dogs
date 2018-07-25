"""Provides interface for training the model."""

import os
import shutil

import tensorflow as tf

import constants as c
from model import architecture
from preprocessing.pipelines import ImageDataPipeline


class TrainingPipeline(ImageDataPipeline):

    def __init__(self,
                 rescale,
                 colorspace,
                 current_bounds,
                 desired_bounds,
                 dtype='float32'):
        """
        TODO
        """
        self.sess = tf.Session()
        ImageDataPipeline.__init__(self, rescale, colorspace, current_bounds, desired_bounds,
                                   dtype)

    def clear_tensorboard(self,
                          path):
        """
        Clears the directory for tensorboard, if it exists.

        Parameters:
            - path (str)
                - Path to the directory.
        """
        if os.path.isdir(path):
            shutil.rmtree(path)

    def build_graph(self):
        """
        Builds a graph for the training process.

        Returns:
            - An input placeholder.
            - A label placeholder.
            - An optimizer.minimize method from TensorFlow.
        """
        input_ = tf.placeholder(tf.float32, shape=c.IN_SHAPE, name='input')
        output = architecture.primary(input_, name='model')
        self.sess.run(tf.global_variables_initializer())
        label = tf.placeholder(tf.float32, shape=list(c.ENCODING.values())[0].shape)
        objective = tf.sqrt(tf.reduce_mean(tf.squared_difference(label, output)), name='objective')
        optimizer = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(objective, name='optimizer')
        tf.summary.scalar('objective_summary', objective)
        return input_, label, optimizer

    def load_graph(self,
                   savemodel_dir):
        """
        Loads an existing graph from a directory saved to by a tf.train.Saver object.

        Parameters:
            - savemodel_dir (str)
                - X/Y
                    - X is the name of the directory that holds saved data about the model.
                    - Y is the prefix for the .data, .index, and .meta files.
        """
        pass

    def initialize_summary_nodes(self,
                                 tensorboard_dir):
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
