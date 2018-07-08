"""Provides interface for frequently used or very important constant values."""

# The rows, columns, and channels of the input-image.
ROWS = 128
COLS = 128
CHAN = 3
# Color space that the input-image will be converted to.
COLOR_SPACE = 'CIELAB'
# Provides `encoding` argument for `preprocessing.ImagePreprocessor.preprocess_directory`.
# WARNING DANGER HAZARD: ANY CHANGE TO THE LABELS MUST BE ACCOUNTED FOR IN `LABEL_SHAPE` BELOW!
ENCODING = {'cats': [1, 0], 'dogs': [0, 1]}
# Shape of a label as shown above in `encoding`.
LABEL_SHAPE = [1, 2]
# Provides `train_dir` argument for `preprocessing.ImagePreprocessor.preprocess_directory`.
TRAIN_DIR = 'data/train'
# X/Y : X is the name of the directory that will hold saved data about the model.
# Y is the prefix for the .data, .index, and .meta files.
SAVEMODEL_DIR = 'saved/model'
# This directory will hold tensorboard files.
TENSORBOARD_DIR = 'tensorboard'
# List of image-formats supported by OpenCV.
SUPPORTED_FORMATS = [
    '.bmp',
    '.pbm',
    '.pgm',
    '.ppm',
    '.sr',
    '.ras',
    '.jpeg',
    '.jpg',
    '.jpe',
    '.jp2',
    '.tiff',
    '.tif',
    '.png'
]
