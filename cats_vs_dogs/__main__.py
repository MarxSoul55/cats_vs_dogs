"""Entry point."""

import argparse
import msvcrt
import sys

from src import pytorch_impl as pyti
from src import tensorflow_impl as tfi


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
    if args.implementation == 'pytorch':
        if args.train:
            training_prompt()
            pyti.src.train.main(pyti.constants.TRAIN_DIR, pyti.constants.ENCODING, args.steps,
                                pyti.constants.SAVEPATH, resuming=args.resuming)
        # TODO
        elif args.classify:
            pass
    # TODO
    elif args.implementation == 'tensorflow':
        pass
