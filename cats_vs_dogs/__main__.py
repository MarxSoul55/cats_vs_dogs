"""Entry point script; implements CLI."""

import argparse
import msvcrt
import sys

from src.pytorch_impl import constants as pyt_constants
from src.pytorch_impl.src import classify as pyt_classify
from src.pytorch_impl.src import train as pyt_train
from src.tensorflow_impl import constants as tf_constants
from src.tensorflow_impl.src import classify as tf_classify
from src.tensorflow_impl.src import train as tf_train


def main():
    pass


if __name__ == '__main__':
    main()
