"""Provides interface for the definition of the model's architecture."""

import tensorflow as tf


def model(input_):
    """
    Builds the model's architecture on the graph.

    # Parameters
        input_ (tf.placeholder): Placeholder for the input data.
    # Returns
        The output of the model.
    """
    op1 = tf.layers.conv2d(input_, 8, 3, padding='same', activation=tf.nn.elu,
                           kernel_initializer=tf.initializers.orthogonal())
    op2 = tf.layers.conv2d(op1, 8, 3, padding='same', activation=tf.nn.elu,
                           kernel_initializer=tf.initializers.orthogonal())
    op3 = tf.layers.conv2d(op2, 8, 3, padding='same', activation=tf.nn.elu,
                           kernel_initializer=tf.initializers.orthogonal())
    op4 = op1 + op3
    op5 = tf.layers.average_pooling2d(op4, 2, 2)
    op6 = tf.layers.conv2d(op5, 16, 3, padding='same', activation=tf.nn.elu,
                           kernel_initializer=tf.initializers.orthogonal())
    op7 = tf.layers.conv2d(op6, 16, 3, padding='same', activation=tf.nn.elu,
                           kernel_initializer=tf.initializers.orthogonal())
    op8 = tf.layers.conv2d(op7, 16, 3, padding='same', activation=tf.nn.elu,
                           kernel_initializer=tf.initializers.orthogonal())
    op9 = op6 + op8
    op10 = tf.layers.average_pooling2d(op9, 2, 2)
    op11 = tf.layers.conv2d(op10, 32, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op12 = tf.layers.conv2d(op11, 32, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op13 = tf.layers.conv2d(op12, 32, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op14 = op11 + op13
    op15 = tf.layers.average_pooling2d(op14, 2, 2)
    op16 = tf.layers.conv2d(op15, 64, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op17 = tf.layers.conv2d(op16, 64, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op18 = tf.layers.conv2d(op17, 64, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op19 = op16 + op18
    op20 = tf.layers.average_pooling2d(op19, 2, 2)
    op21 = tf.layers.conv2d(op20, 128, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op22 = tf.layers.conv2d(op21, 128, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op23 = tf.layers.conv2d(op22, 128, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op24 = op21 + op23
    op25 = tf.layers.average_pooling2d(op24, 2, 2)
    op26 = tf.layers.conv2d(op25, 256, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op27 = tf.layers.conv2d(op26, 256, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op28 = tf.layers.conv2d(op27, 256, 3, padding='same', activation=tf.nn.elu,
                            kernel_initializer=tf.initializers.orthogonal())
    op29 = op26 + op28
    op30 = tf.layers.average_pooling2d(op29, 2, 2)
    op31 = tf.layers.average_pooling2d(op30, op30.shape.as_list()[1], 1)
    op32 = tf.reshape(op31, [1, op31.shape.as_list()[3]], name='refined')
    op33 = tf.layers.dense(op32, 2, kernel_initializer=tf.initializers.orthogonal())
    output = tf.identity(op33, name='output')
    return output
