"""Entry point script; implements CLI."""

import argparse
import msvcrt
import pickle
import sys

from PyInquirer import prompt

from src.pytorch_impl import constants as pyt_constants
from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train
from src.tensorflow_impl import constants as tf_constants
from src.tensorflow_impl.src import classify as tf_classify
from src.tensorflow_impl.src import train as tf_train


def yes_or_no(message):
    """
    Prompts the user with a custom yes-or-no prompt.

    Parameters:
        - message (str)
            - Message to be printed to the user.
    Returns:
        - A string; either 'y' or 'n'.
    """
    print(message)
    while True:
        resp = msvcrt.getch().decode().lower()
        if resp == 'y' or resp == 'n':
            return resp
        else:
            print('Press either the Y or N key.')


def save_config(train_dir,
                encoding,
                savepath):
    """
    Saves dataset and model information to disk.

    Parameters:
        - train_dir (str)
            - Path to the directory of classes.
            - May be relative or absolute.
            - e.g. 'data/train' (where 'train' holds the subdirs)
        - encoding (dict, str --> np.ndarray)
            - Maps the name of the subdirectory (class) to a label.
                - e.g. {'cats': np.array([[1, 0]]), 'dogs': np.array([[0, 1]])}
                    - Each label must have the same shape!
                    - In this case, the two labels are of shape [1, 2].
        - savepath
            - See the `src.train` module in the specific implementation for details.

    """
    data = {
        'train_dir': train_dir,
        'encoding': encoding,
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
    operation_question = [
        {
            'type': 'list',
            'name': 'selected_operation',
            'message': 'Select operation to perform:',
            'choices': [
                'Train model on dataset.',
                'Test model on testset.',
                'Classify an image or directory of images.'
            ]
        }
    ]
    selected_operation = prompt(operation_question)['selected_operation']
    if selected_operation == 'Train model on dataset.':
        pass
    elif selected_operation == 'Test model on testset.':
        pass
    elif selected_operation == 'Classify an image or directory of images.':
        pass


if __name__ == '__main__':
    main()
