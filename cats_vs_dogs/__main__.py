"""Entry point script; implements CLI."""

import sys

from pyfiglet import Figlet
from PyInquirer import prompt

from src.pytorch_impl import constants as pyt_constants
from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train

# User must select operation first.
OPERATION_QUESTION = [
    {
        'type': 'list',
        'name': 'selected_operation',
        'message': 'Select operation to perform:',
        'choices': [
            'Train model on dataset.',
            'Test model on testset.',
            'Classify an image or directory of images.',
            'Exit.'
        ]
    }
]
# User wants to train? Setup all relevant arguments.
TRAINING_MENU = [
    {
        'type': 'list',
        'name': 'resuming',
        'message': 'Are you training from scratch, or from a saved model?',
        'choices': [
            'Training from scratch.',
            'Training from a saved model.'
        ]
    },
    {
        'type': 'input',
        'name': 'train_dir',
        'message': 'Enter the path to the directory that contains the subdirs (classes):'
    },
    {
        'type': 'input',
        'name': 'steps',
        'message': 'Enter number of steps (gradient updates of SGD) to perform: (e.g. 1000)',
        'filter': lambda x: int(x)
    },
    {
        'type': 'input',
        'name': 'savepath',
        'message': 'Enter path to save trained model to: (e.g. X/Y.pth)'
    }
]


def main():
    print(Figlet(font='slant').renderText('cats_vs_dogs'))
    selected_operation = prompt(OPERATION_QUESTION)['selected_operation']
    if selected_operation == 'Train model on dataset.':
        train_vars = prompt(TRAINING_MENU)
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
