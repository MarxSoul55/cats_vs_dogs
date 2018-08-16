"""Entry point script; implements CLI."""

import argparse
import msvcrt
import pickle
import sys

from PyInquirer import prompt

import questions
from src.pytorch_impl import constants as pyt_constants
from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train
from src.tensorflow_impl import constants as tf_constants
from src.tensorflow_impl.src import classify as tf_classify
from src.tensorflow_impl.src import train as tf_train


def save_config(train_dir,
                encoding,
                savepath):
    """
    Saves dataset and model information to disk.

    Parameters:
        - train_dir, encoding
            - See `preprocessing.ImageDataPipeline.preprocess_classes` in either impl for details.
        - savepath
            -

    """
    pass


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
