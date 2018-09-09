"""EDIT ME TO CHANGE THE PROGRAM'S CONFIGURATION!"""

# DO NOT REMOVE THIS IMPORT — NUMPY IS NEEDED FOR LABEL_DICT!
import numpy as np

# This directory should contain subdirectories which each contain a bunch of images.
# The names of those subdirectories should correspond to the keys for LABEL_DICT.
TRAIN_DIR = 'cats_vs_dogs/data/train'
# This constant maps the names of the subdirectories in TRAIN_DIR to tensor labels.
# For obvious reasons, it must have the same
LABEL_DICT = {
    'cat': np.array([[1, 0]], dtype='float32'),
    'dog': np.array([[0, 1]], dtype='float32')
}
# This constant controls where the saved model is located.
# The model will also be loaded from this area for resuming training and classification.
# If using PyTorch version: path to a .pth file. This file encapsulates the saved model.
# If using TensorFlow version: path to a directory. TensorFlow uses multiple files to save a model.
SAVEPATH = 'cats_vs_dogs/bin/pytorch_impl/babyresnet.pth'
# ONLY NEEDED FOR TENSORFLOW VERSION!
# This controls what directory TensorBoard information will be saved to.
TENSORBOARD_DIR = 'cats_vs_dogs/bin/tensorflow_impl/tensorboard'
