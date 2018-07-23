"""Provides interface for frequently used or very important constant values."""

import numpy as np

# Shape of the input in NHWC format.
IN_SHAPE = [1, 128, 128, 3]
# Color space that the input-image will be converted to.
COLORSPACE = 'CIELAB'
# Provides `encoding` argument for `preprocessing.ImagePreprocessor.preprocess_classes`.
# ALL LABELS MUST BE RANK 1 TENSORS! (i.e. shape=[1, X] where X is some integer)
# For obvious reasons, the labels' shapes must also match the model's output shape.
ENCODING = {'cats': np.array([[1, 0]]), 'dogs': np.array([[0, 1]])}
# Provides `train_dir` argument for `preprocessing.ImagePreprocessor.preprocess_classes`.
TRAIN_DIR = '../data/train'
# X/Y : X is the name of the directory that will hold saved data about the model.
# Y is the prefix for the .data, .index, and .meta files.
SAVEMODEL_DIR = '../bin/saved/model'
# This directory will hold tensorboard files.
TENSORBOARD_DIR = '../bin/tensorboard'
