"""Defines (and provides a builder-function for) the architecture of the model."""

import tensorflow as tf


def model(input_):
    """
    Builds the model's architecture on the graph.

    # Parameters
        input_ (tf.placeholder): Placeholder for the input data.
    # Returns
        The output of the model.
    """
    op1 = tf.layers.conv2d(input_, 10, 5, padding='same', activation=tf.nn.relu,
                           kernel_initializer=tf.initializers.orthogonal())
    op2 = tf.layers.conv2d(op1, 10, 5, padding='same', activation=tf.nn.relu,
                           kernel_initializer=tf.initializers.orthogonal())
    op3 = tf.layers.conv2d(op2, 10, 5, padding='same', activation=tf.nn.relu,
                           kernel_initializer=tf.initializers.orthogonal())
    op4 = op1 + op3
    op5 = tf.layers.average_pooling2d(op4, 2, 2)
    op6 = tf.layers.conv2d(op5, 20, 5, padding='same', activation=tf.nn.relu,
                           kernel_initializer=tf.initializers.orthogonal())
    op7 = tf.layers.conv2d(op6, 20, 5, padding='same', activation=tf.nn.relu,
                           kernel_initializer=tf.initializers.orthogonal())
    op8 = tf.layers.conv2d(op7, 20, 5, padding='same', activation=tf.nn.relu,
                           kernel_initializer=tf.initializers.orthogonal())
    op9 = op6 + op8
    op10 = tf.layers.average_pooling2d(op9, 2, 2)
    op11 = tf.layers.conv2d(op10, 40, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op12 = tf.layers.conv2d(op11, 40, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op13 = tf.layers.conv2d(op12, 40, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op14 = op11 + op13
    op15 = tf.layers.average_pooling2d(op14, 2, 2)
    op16 = tf.layers.conv2d(op15, 80, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op17 = tf.layers.conv2d(op16, 80, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op18 = tf.layers.conv2d(op17, 80, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op19 = op16 + op18
    op20 = tf.layers.average_pooling2d(op19, 2, 2)
    op21 = tf.layers.conv2d(op20, 160, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op22 = tf.layers.conv2d(op21, 160, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op23 = tf.layers.conv2d(op22, 160, 5, padding='same', activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.orthogonal())
    op24 = op21 + op23
    op25 = tf.layers.average_pooling2d(op24, 2, 2)
    op26 = tf.layers.average_pooling2d(op25, op25.shape.as_list()[1], 1)
    op27 = tf.reshape(op26, [1, op26.shape.as_list()[3]])
    op28 = tf.layers.dense(op27, 2, kernel_initializer=tf.initializers.orthogonal())
    output = tf.identity(op28, name='output')
    return output
