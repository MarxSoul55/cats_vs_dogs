"""Provides interface for the definition of the model's architecture."""

import tensorflow as tf


def model(input_, name=None):
    """
    Builds the model's architecture on the graph.

    # Parameters
        input_ (tf.placeholder):
            - Placeholder for the input data.
        name (str):
            - The name of the scope of the model's operations.
    # Returns
        - The output of the model.
    """
    with tf.name_scope(name):
        skip = tf.layers.conv2d(input_, 8, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(skip, 8, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(x, 8, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(skip, 16, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(skip, 32, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(skip, 64, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 128, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(skip, 128, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(x, 128, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 256, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(skip, 256, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = tf.layers.conv2d(x, 256, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.initializers.orthogonal())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.average_pooling2d(x, x.shape.as_list()[1], 1)
        x = tf.reshape(x, [1, x.shape.as_list()[3]], name='distilled')
        x = tf.layers.dense(x, 2, kernel_initializer=tf.initializers.orthogonal())
        x = tf.identity(x, name='output')
        return x
