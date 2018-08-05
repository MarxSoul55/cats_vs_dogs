"""Provides interface for training the model."""

import os
import shutil

import tensorflow as tf

from .model import architecture
from .preprocessing import ImageDataPipeline


def train(train_dir,
          encoding,
          steps,
          savepath,
          tensorboard_dir,
          resuming=True):
    """
    Builds up a graph of operations from scratch to train a model.

    Parameters:
        - train_dir, encoding
            - Parameters for the preprocessor.
            - See `preprocessing.ImageDataPipeline.preprocess_classes` for details.
        - steps (int)
            - Number of gradient updates (samples to train on).
        - savepath (str)
            - Path where the model will be saved/loaded from.
            - e.g. X/Y will give: X/Y.meta, X/Y.index, etc. for other tf.train.Saver artifacts.
        - tensorboard_dir (str)
            - Directory where the tensorboard files are stored.
        - resuming (bool)
            - Whether to resume training from a saved model or to start from scratch.
    """
    if os.path.isdir(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
    sess = tf.Session()
    if resuming:
        loader = tf.train.import_meta_graph(savepath + '.meta')
        loader.restore(sess, savepath)
        input_ = sess.graph.get_tensor_by_name('input:0')
        label = sess.graph.get_tensor_by_name('label:0')
        optimizer = sess.graph.get_operation_by_name('optimizer')
    else:
        input_ = tf.placeholder(tf.float32, name='input')
        output = architecture.baby_resnet(input_, name='model')
        sess.run(tf.global_variables_initializer())
        label = tf.placeholder(tf.float32, name='label')
        objective = tf.sqrt(tf.reduce_mean(tf.squared_difference(label, output)), name='objective')
        optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(objective, name='optimizer')
        tf.summary.scalar('objective_summary', objective)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, graph=sess.graph)
    preprocessor = ImageDataPipeline()
    for step, img_path, img_tensor, img_label in preprocessor.preprocess_classes(steps, train_dir,
                                                                                 encoding):
        print('Step: {} | Image: {}'.format(step, img_path))
        _, step_summary = sess.run([optimizer, summary],
                                   feed_dict={input_: img_tensor, label: img_label})
        writer.add_summary(step_summary, global_step=step)
    tf.train.Saver().save(sess, savepath)
    print('\a')
