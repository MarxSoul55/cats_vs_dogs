"""Entry point script; implements CLI."""

import argparse
import msvcrt
import pickle
import sys

from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train
from src.tensorflow_impl.src import classify as tf_classify
from src.tensorflow_impl.src import train as tf_train


def training_prompt():
    """
    Prompts the user with a warning message about overwriting the saved model.
    """
    print('WARNING: Training will overwrite the saved model (if it exists). EXECUTE Y/N?')
    while True:
        resp = msvcrt.getch().decode().lower()
        if resp == 'y':
            print('Loading...')
            return
        elif resp == 'n':
            sys.exit('Training aborted.')
        else:
            print('Press either the Y or N key.')


def print_prediction(prediction):
    """
    Prints the predicted label(s) by the model to the screen.

    Parameters:
        - prediction (str or dict)
            - Prediction(s) generated by model.
            - Type depends on whether a single image or a directory of images was classified.
    """
    if type(prediction) == str:
        print(prediction)
    elif type(prediction) == dict:
        keys, values = list(prediction.keys()), list(prediction.values())
        for key, value in zip(keys, values):
            print(key, value)


def parse_arguments():
    """
    Parses the CLI for specific arguments via the `argparse` library.

    Returns:
        - A dictionary mapping each argument to the given value.
        - Each string value is lowercased to prevent case sensitivity issues.
    """
    # Specification for CLI args.
    description = 'See for help: https://github.com/MarxSoul55/cats_vs_dogs/wiki'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resuming', action='store_true')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--label_dict_path', type=str)
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--tensorboard_dir', type=str)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--source', type=str)
    parser.add_argument('--implementation', type=str)
    parser.set_defaults(train=False,
                        resuming=False,
                        train_dir=None,
                        label_dict_path=None,
                        savepath=None,
                        tensorboard_dir=None,
                        steps=None,
                        classify=False,
                        source=None,
                        implementation='pytorch')
    # Parse args into dict, and load label_dict from given path.
    args = vars(parser.parse_args())
    with open(args['label_dict_path'], 'rb') as f:
        args['label_dict'] = pickle.load(f)
        del args['label_dict_path']
    return args


def main(args):
    """
    Executes the program.

    Parameters:
        - args (dict, str -> ?)
            - A dictionary converted from an `argparse.Namespace` object.
            - Maps CLI arguments to their values.
    """
    if args['train']:
        training_prompt()
        if args['implementation'] == 'pytorch':
            pyt_train.main(args['train_dir'], args['label_dict'], args['steps'], args['savepath'],
                           resuming=args['resuming'])
        elif args['implementation'] == 'tensorflow':
            tf_train.main(args['train_dir'], args['label_dict'], args['steps'], args['savepath'],
                          args['tensorboard_dir'], resuming=args['resuming'])
    elif args['classify']:
        if args['implementation'] == 'pytorch':
            prediction = pyt_classify.main(args['source'], args['savepath'], args['label_dict'])
        elif args['implementation'] == 'tensorflow':
            prediction = tf_classify.main(args['source'], args['savepath'], args['label_dict'])
        print_prediction(prediction)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
