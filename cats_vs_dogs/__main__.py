"""Entry point script; implements CLI."""

import argparse
import msvcrt
import os
import pickle
import sys

from PyInquirer import prompt

import questions
from src.pytorch_impl import constants as pyt_constants
from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train


def save_config(train_dir,
                savepath):
    """
    Saves dataset and model information to disk.

    Parameters:
        - train_dir (str)
            - Path to the directory of classes.
            - e.g. 'data/train', where 'train' holds subdirs with images in them.
            - IMPORTANT: This directory should also hold a file called `label.pkl`.
                - A (pickled) dictionary of numpy arrays.
                - Maps the name of the subdirectory (class) to a label.
                    - e.g. {'cats': np.array([[1, 0]]), 'dogs': np.array([[0, 1]])}
                        - Each label must have the same shape!
                        - In this case, the two labels are of shape [1, 2].
        - savepath
            - See the `src.train` module in the specific implementation for details.
    """
    data = {
        'train_dir': train_dir,
        'savepath': savepath
    }
    with open('config.pkl') as f:
        pickle.dump(data, f)


def load_config():
    """
    Loads configuration data from `config.pkl`.

    Returns:
        - A dictionary with keys for 'train_dir', 'encoding', and 'savepath'.
    """
    with open('config.pkl') as f:
        return pickle.load(f)


def main():
    if 'config.pkl' in os.listdir():
        use_saved_config = prompt(questions.CONFIGURATION_QUESTION)['use_saved_config']
        if use_saved_config:
            config = load_config()
        else:
            config = {}
    else:
        config = {}
    selected_operation = prompt(questions.OPERATION_QUESTION)['selected_operation']
    if selected_operation == 'Train model on dataset.':
        if len(config) == 0:
            config = prompt(questions.TRAINING_PATH_QUESTIONS)
            print('These paths will be saved in `config.pkl`.')
            save_config(config['train_dir'], config['savepath'])
        training_variables = prompt(questions.TRAINING_VARIABLE_QUESTIONS)
        pyt_train(config['train_dir'], training_variables['steps'], config['savepath'],
                  resuming=training_variables['resuming'])
    elif selected_operation == 'Test model on testset.':
        pass
    elif selected_operation == 'Classify an image or directory of images.':
        pass


if __name__ == '__main__':
    main()
