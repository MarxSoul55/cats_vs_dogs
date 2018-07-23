"""Entry point."""

import argparse
import msvcrt
import sys

from src.classify import classify
from src.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Need help? See: https://github.com/MarxSoul55/cats_vs_dogs#using-the-cli')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resuming', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--source')
    parser.set_defaults(resuming=False)
    args = parser.parse_args()
    if args.train:
        print('WARNING: Training will overwrite the saved model (if it exists). EXECUTE Y/N?')
        if msvcrt.getch().decode().lower() == 'y':
            train(args.steps, args.resuming)
        else:
            sys.exit('Program closed.')
    elif args.classify:
        print(classify(args.source))
