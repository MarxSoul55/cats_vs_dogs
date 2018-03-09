"""Provides interface for constant values."""

# Size of the minibatch when training.
BATCH = 1
# The rows, columns, and channels of the input-image.
ROWS = 256
COLS = 256
CHAN = 3
# Provides `encoding` argument for `preprocessing.ImagePreprocessor.preprocess_directory`.
ENCODING = {'cats': [1, 0], 'dogs': [0, 1]}
# Provides `train_dir` argument for `preprocessing.ImagePreprocessor.preprocess_directory`.
TRAIN_DIR = 'data/train'
# This directory will hold saved data about the model.
SAVEMODEL_DIR = 'saved/model'
# This directory will hold tensorboard files.
TENSORBOARD_DIR = 'tensorboard'
