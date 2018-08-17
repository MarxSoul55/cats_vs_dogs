"""Entry point script; implements CLI."""

import argparse
import msvcrt
import pickle
import sys

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
    train_help = '(flag) Tells program to train the model.'
    resuming_help = ('(flag) Tells program to resume training off of a saved model '
                     'whose path is given by the --savepath arg.')
    train_dir_help = ('(str) Tells program where training dataset is; this directory should '
                      'contain subdirectories (representing each class) of images.')
    label_dict_path_help = ('(str) This path should point to a .pkl file of a dictionary mapping '
                            'strings (specifically, the names of the subdirectories in the '
                            '--train_dir arg) to numpy arrays. The shape of each label should '
                            'be the same and it should match the output shape of the model.')
    savepath_help = ('(str) Path that indicates where the model will be saved and loaded from. '
                     'File extension should be: .pth')
    steps_help = '(int) Indicates how many images to train on (one gradient update per image).'
    classify_help = ('(flag) Tells program to classify something using the saved '
                     'model from the --savepath arg.')
    source_help = '(str) A path to either an image or directory of images to classify.'
    implementation_help = ('(str) Either \'pytorch\' (default) or \'tensorflow\'. Note that '
                           'the TensorFlow implementation uses a `constants` module to hold '
                           'important paths instead of the CLI args for `train_dir`, '
                           '`label_dict_path`, and `savepath`. SETTING THESE ARGS FOR THE '
                           'TENSORFLOW IMPLEMENTATION IS POINTLESS!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help=train_help)
    parser.add_argument('--resuming', action='store_true', help=resuming_help)
    parser.add_argument('--train_dir', type=str, help=train_dir_help)
    parser.add_argument('--label_dict_path', type=str, help=label_dict_path_help)
    parser.add_argument('--savepath', type=str, help=savepath_help)
    parser.add_argument('--steps', type=int, help=steps_help)
    parser.add_argument('--classify', action='store_true', help=classify_help)
    parser.add_argument('--source', type=str, help=source_help)
    parser.add_argument('--implementation', type=str, help=implementation_help)
    parser.set_defaults(train=False,
                        resuming=False,
                        train_dir=None,
                        label_dict_path=None,
                        savepath=None,
                        steps=None,
                        classify=False,
                        source=None,
                        implementation='pytorch')
    args = vars(parser.parse_args())
    with open(args['label_dict_path'], 'rb') as f:
        args['label_dict'] = pickle.load(f)
        del args['label_dict_path']
    return args


def main(args):
    """
    Executes the program.

    Parameters:
        - args (dict)
            - A dictionary converted from an argparse.Namespace object.
    """
    if args['implementation'] == 'pytorch':
        if args['train']:
            training_prompt()
            pyt_train.main(args['train_dir'], args['label_dict'], args['steps'],
                           args['savepath'], resuming=args['resuming'])
        elif args['classify']:
            prediction = pyt_classify.main(args['source'], args['savepath'],
                                           args['label_dict'])
            print_prediction(prediction)
    elif args['implementation'] == 'tensorflow':
        if args['train']:
            training_prompt()
            tf_train.main(tf_constants.TRAIN_DIR, tf_constants.ENCODING, args['steps'],
                          tf_constants.SAVEPATH, tf_constants.TENSORBOARD_DIR,
                          resuming=args['resuming'])
        elif args['classify']:
            prediction = tf_classify.main(args['source'], tf_constants.SAVEPATH,
                                          tf_constants.ENCODING)
            print_prediction(prediction)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
