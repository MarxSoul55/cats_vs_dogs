"""Entry point."""

import argparse
import msvcrt
import sys

from src.pytorch_impl import constants as pyt_constants
from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train


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
            - An object returned from `argparse.ArgumentParser().parse_args()`.
    """
    if args.implementation == 'pytorch':
        if args.train:
            training_prompt()
            pyt_train.main(pyt_constants.TRAIN_DIR, pyt_constants.ENCODING, args.steps,
                           pyt_constants.SAVEPATH, resuming=args.resuming)
        elif args.classify:
            predicted_label = pyt_classify.main(args.source, pyt_constants.SAVEPATH,
                                                pyt_constants.ENCODING)
            print(predicted_label)
    # TODO
    elif args.implementation == 'tensorflow':
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Need help? See: https://github.com/MarxSoul55/cats_vs_dogs#using-the-cli')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resuming', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--source', type=str)
    parser.add_argument('--implementation', type=str)
    parser.set_defaults(resuming=False, implementation='pytorch')
    args = parser.parse_args()
    main(args)
