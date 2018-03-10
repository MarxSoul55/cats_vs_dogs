"""Provides interface for constant values."""

# Size of the minibatch when training.
BATCH = 1
# The rows, columns, and channels of the input-image.
ROWS = 512
COLS = 512
CHAN = 3
# Provides `encoding` argument for `preprocessing.ImagePreprocessor.preprocess_directory`.
# WARNING DANGER HAZARD: ANY CHANGE TO THE LABELS MUST BE ACCOUNTED FOR IN `LABEL_SHAPE` BELOW!
ENCODING = {'cats': [1, 0], 'dogs': [0, 1]}
# Shape of a label as shown above in `encoding`.
LABEL_SHAPE = [1, 2]
# Provides `train_dir` argument for `preprocessing.ImagePreprocessor.preprocess_directory`.
TRAIN_DIR = 'data/train'
# This directory will hold saved data about the model.
SAVEMODEL_DIR = 'saved/model'
# This directory will hold tensorboard files.
TENSORBOARD_DIR = 'tensorboard'
