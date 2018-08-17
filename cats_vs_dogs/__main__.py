"""Entry point script; implements CLI."""

import sys

from pyfiglet import Figlet
from PyInquirer import prompt

import questions
from src.pytorch_impl import constants as pyt_constants
from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train


def main():
    print(Figlet(font='slant').renderText('cats_vs_dogs'))
    selected_operation = prompt(questions.OPERATION_QUESTION)['selected_operation']
    if selected_operation == 'Train model on dataset.':
        train_vars = prompt(questions.TRAINING_MENU)
        resuming = True if train_vars['resuming'] == 'Training from a saved model.' else False
        pyt_train.main(train_vars['train_dir'], train_vars['steps'], train_vars['savepath'],
                       resuming=resuming)
    elif selected_operation == 'Test model on testset.':
        pass
    elif selected_operation == 'Classify an image or directory of images.':
        pass
    elif selected_operation == 'Exit.':
        sys.exit()


if __name__ == '__main__':
    main()
