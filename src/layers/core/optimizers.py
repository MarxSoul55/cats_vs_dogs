"""Provides interface for optimizers."""

import tensorflow as tf


def adadelta(objective, learning_rate=1.0, rho=0.95, epsilon=1E-8):
    """
    Performs gradient-descent with the adadelta method.

    # Parameters
        objective (operation): Objective function being used.
        learning_rate (float): Cuts the gradient.
        rho (float): Hyperparameter as in the paper.
        epsilon (float): Number to add to denominator to avoid division by zero.
    # Returns
        An optimization operation.
    """
    return tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=rho,
                                      epsilon=epsilon).minimize(objective)


def adam(objective, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1E-8):
    """
    Performs gradient-descent with the adam method.

    # Parameters
        objective (operation): Objective function being used.
        learning_rate (float): Cuts the gradient.
        beta1 (float): Hyperparameter as in the paper.
        beta2 (float): Hyperparameter as in the paper.
        epsilon (float): Number to add to denominator to avoid division by zero.
    # Returns
        An optimization operation.
    """
    return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                  epsilon=epsilon).minimize(objective)


def momentum(objective, learning_rate=0.001, discount=0.9):
    """
    Performs gradient-descent with momentum.

    # Parameters
        objective (operation): Objective function being used.
        learning_rate (float): Cuts the gradient.
        discount (float): Frictional coefficient that slows momentum.
    # Returns
        An optimization operation.
    """
    return tf.train.MomentumOptimizer(learning_rate, discount,
                                      use_nesterov=False).minimize(objective)


def nesterov_momentum(objective, learning_rate=0.001, discount=0.9):
    """
    Performs gradient-descent with nesterov-momentum.

    # Parameters
        objective (operation): Objective function being used.
        learning_rate (float): Cuts the gradient.
        discount (float): Frictional coefficient that slows momentum.
    # Returns
        An optimization operation.
    """
    return tf.train.MomentumOptimizer(learning_rate, discount,
                                      use_nesterov=True).minimize(objective)


def rmsprop(objective, learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1E-10):
    """
    Performs gradient-descent with RMSProp.

    # Parameters
        objective (operation): Objective function being used.
        learning_rate (float): Cuts the gradient.
        decay (float): Hyperparameter as in the paper.
        momentum (float): Hyperparameter as in the paper.
        epsilon (float): Number to add to denominator to avoid division by zero.
    # Returns
        An optimization operation.
    """
    return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum,
                                     epsilon=epsilon).minimize(objective)
