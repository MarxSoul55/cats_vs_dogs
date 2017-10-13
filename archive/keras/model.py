"""Convolutional Neural Networks MEET... Cats & Dogs!"""

import argparse
import math
import os

import numpy as np
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Dimensions (rows, columns) of the input images. If not this size, will be resized to it.
DIM = 128


def train(epochs, resuming):
    """
    Trains the network and saves its configuration into a .h5 file.

    # Parameters
        epochs (int): Number of epochs to train the model.
        resuming (bool): Whether to resume training from a .h5 file, or to train from scratch.
    """
    if resuming:
        model = load_model('saved_model.h5')
        model.summary()
    else:
        nn_config = [
            Conv2D(32, 7, strides=2, padding='same', input_shape=(DIM, DIM, 1)),
            BatchNormalization(),
            ELU(),
            MaxPooling2D(pool_size=2, strides=2, padding='valid'),
            Conv2D(64, 3, strides=1, padding='same'),
            BatchNormalization(),
            ELU(),
            Conv2D(64, 3, strides=1, padding='same'),
            BatchNormalization(),
            ELU(),
            MaxPooling2D(pool_size=2, strides=2, padding='valid'),
            Conv2D(128, 3, strides=1, padding='same'),
            BatchNormalization(),
            ELU(),
            Conv2D(128, 3, strides=1, padding='same'),
            BatchNormalization(),
            ELU(),
            Conv2D(128, 3, strides=1, padding='same'),
            BatchNormalization(),
            ELU(),
            MaxPooling2D(pool_size=2, strides=2, padding='valid'),
            GlobalAveragePooling2D(),
            Dense(1, activation='sigmoid')
        ]
        model = Sequential(nn_config)
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    data_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    train_gen = data_gen.flow_from_directory('data/train', target_size=(DIM, DIM), batch_size=32,
                                             classes=['cats', 'dogs'], class_mode='binary',
                                             color_mode='grayscale')
    model.fit_generator(train_gen, math.ceil(25000/32), epochs=epochs)
    print('Saving model...')
    model.save('saved_model.h5')
    print('Model saved!')


def classify_dir(folder_path):
    """
    Given a path to a directory, classifies all the images inside it
    (whose formats are supported by PIL).

    # Parameters
        folder_path (str): Path to a directory.
    # Returns
        A dictionary of classifications, images' filenames as keys.
    # Raises
        FileNotFoundError: if the function found no images.
    """
    folder = os.listdir(folder_path)
    image_filenames = []  # Used for creating dictionary at end of function.
    classifier_inputs = []
    for item in folder:
        path = folder_path + '/' + item
        if os.path.isdir(path):
            continue
        try:
            img = load_img(path, grayscale=True, target_size=(DIM, DIM))
            image_filenames.append(item)
        except:  # Format isn't supported by PIL.
            continue
        img = img_to_array(img)
        img -= np.mean(img)
        img /= np.std(img)
        classifier_inputs.append(img)
    if len(classifier_inputs) == 0:
        raise FileNotFoundError('No images found (whose formats are supported by PIL).')
    classifier_inputs = np.array(classifier_inputs)
    model = load_model('saved_model.h5')
    preds = model.predict_classes(classifier_inputs, batch_size=32, verbose=0)
    dict_of_preds = {}
    for name, pred in zip(image_filenames, preds):
        if pred[0] == 0:
            dict_of_preds[name] = 'cat'
        else:
            dict_of_preds[name] = 'dog'
    return dict_of_preds


def classify_img(img_path):
    """
    Classify a single image (whose format is supported by PIL).

    # Parameters
        img_path (str): Path to the image, including file-extension.
    # Raises
        FileNotFoundError: if `img_path` is a directory.
        FileNotFoundError: if image doesn't exist or format not supported by PIL, or user neglected
                           file-extension.
    # Returns
        A string, either 'cat' or 'dog'.
    """
    if os.path.isdir(img_path):
        raise FileNotFoundError('This is a directory, you goof-ball.')
    try:
        img = load_img(img_path, grayscale=True, target_size=(DIM, DIM))
    except:
        raise FileNotFoundError('Image either doesn\'t exist or format not supported by PIL. '
                                'Also possible that you simply neglected the file-extension.')
    img = img_to_array(img)
    img = img * (1/255)
    img = np.array([img])
    model = load_model('saved_model.h5')
    pred = model.predict_classes(img, batch_size=1, verbose=0)[0][0]
    if pred == 0:
        return 'cat'
    return 'dog'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-tr', '--trainresume', action='store_true')
    parser.add_argument('-cd', '--classifydirectory', action='store_true')
    parser.add_argument('-ci', '--classifyimage', action='store_true')
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-p', '--path')
    args = parser.parse_args()
    if args.train:
        train(args.epochs, False)
    elif args.trainresume:
        train(args.epochs, True)
    elif args.classifydirectory:
        classify_dir(args.path)
    elif args.classifyimage:
        classify_img(args.path)
    else:
        raise Exception('Must add command-line args.')
