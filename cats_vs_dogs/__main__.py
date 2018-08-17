"""Entry point."""

import argparse
import msvcrt
import sys

from src.pytorch_impl import constants as pyt_constants
from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train
from src.tensorflow_impl import constants as tf_constants
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
            return
        elif resp == 'n':
            sys.exit('Training aborted.')
        else:
            print('Press either the Y or N key.')


def main(args):
    """
    Executes the program.

    Parameters:
        - args (argparse.Namespace)
            - An object returned from `argparse.ArgumentParser.parse_args`.
    """
    if args.implementation == 'pytorch':
        if args.train:
            training_prompt()
            pyt_train.main(pyt_constants.TRAIN_DIR, pyt_constants.ENCODING, args.steps,
                           pyt_constants.SAVEPATH, resuming=args.resuming)
        elif args.classify:
            prediction = pyt_classify.main(args.source, pyt_constants.SAVEPATH,
                                           pyt_constants.ENCODING)
            if type(prediction) == str:
                print(prediction)
            elif type(prediction) == dict:
                keys, values = list(prediction.keys()), list(prediction.values())
                for key, value in zip(keys, values):
                    print(key, value)
    elif args.implementation == 'tensorflow':
        if args.train:
            training_prompt()
            tf_train.main(tf_constants.TRAIN_DIR, tf_constants.ENCODING, args.steps,
                          tf_constants.SAVEPATH, tf_constants.TENSORBOARD_DIR,
                          resuming=args.resuming)
        elif args.classify:
            prediction = tf_classify.main(args.source, tf_constants.SAVEPATH,
                                          tf_constants.ENCODING)
            if type(prediction) == str:
                print(prediction)
            elif type(prediction) == dict:
                keys, values = list(prediction.keys()), list(prediction.values())
                for key, value in zip(keys, values):
                    print(key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resuming', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--source', type=str)
    parser.add_argument('--implementation', type=str)
    parser.set_defaults(resuming=False, implementation='pytorch')
    args = parser.parse_args()
    main(args)
