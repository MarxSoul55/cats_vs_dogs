from PyInquirer import prompt, Separator

questions = [
    {
        'type': 'list',
        'name': 'selected_operation',
        'message': 'Select operation:',
        'choices': [
            'Train model on dataset.',
            'Test model on testset.',
            'Classify an image or directory of images.'
        ],
    }
]
answers = prompt(questions)
if answers['selected_operation'] == 'Classify an image or directory of images.':
    print('Hello World!')
