"""Holds constants for PyInquirer's prompt function."""

# What operation does the user wish to perform?
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
# TODO: Clarify train_dir_message for path requirements (needs to be rel. to repo root)
# TODO: Add deeper explanation for train_dir in CLI???
# TODO: Does PyTorch need the savepath to already exist???
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
