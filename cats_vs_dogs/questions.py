"""Holds constants for PyInquirer's prompt function."""
# Does a configuration exist? If so, call this question.
CONFIGURATION_QUESTION = [
    {
        'type': 'confirm',
        'name': 'use_saved_config',
        'message': 'A configuration for this project has been found. Use it?'
    }
]
# What operation does the user wish to perform?
OPERATION_QUESTION = [
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
# User wants to train? Set up all relevant variables.
# TODO: Add deeper explanation for train_dir in CLI???
train_dir_message = ('Enter location of dataset; either absolute path or relative to repo root. '
                     '(e.g. X/Y where Y holds class1_dir_pics, class2_dir_pics, etc.)')
TRAINING_QUESTIONS = [
    {
        'type': 'input',
        'name': 'train_dir',
        'message': train_dir_message
    }
]
