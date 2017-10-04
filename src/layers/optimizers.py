"""Provides interface for optimizers."""

import tensorflow as tf


def momentum(objective, learning_rate=0.01, discount=0.9):
    """
    Performs gradient-descent with momentum.

    # Parameters
        learning_rate (float): Cuts the gradient.
        discount (float): Frictional coefficient that slows momentum.
        objective (operation): Objective function being used.
    # Returns
        An optimization operation.
    """
    return tf.train.MomentumOptimizer(learning_rate, discount).minimize(objective)
