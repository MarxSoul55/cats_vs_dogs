"""Provides interface for training the model."""

import os
import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from . import models
from .preprocessing import ImageDataPipeline


def save(model, path):
    """
    Saves the model.

    Parameters:
        - model (class def)
            - Model definition to save.
        - path (str)
            - Path where the model will be saved to.
            - e.g. hello/world/saved_model_file.pth
    """
    # PyTorch requires parent directory of savepath to exist. Ensure it does.
    parentdir = pathlib.Path(path).parent
    if not os.path.exists(parentdir):
        os.makedirs(parentdir)
    torch.save(model.state_dict(), path)


def main(train_dir,
         label_dict,
         steps,
         savepath,
         resuming=True):
    """
    Trains the model and saves the result.

    Parameters:
        - train_dir (str)
            - Path to the directory of classes.
            - e.g. 'data/train', where 'train' holds subdirs with images in them.
        - label_dict (dict, str -> np.ndarray)
            - Maps the name of the subdirectory (class) to a label.
                - e.g. {'cats': np.array([[1, 0]]), 'dogs': np.array([[0, 1]])}
                    - Each label must have the same shape!
                    - In this case, the two labels are of shape [1, 2].
        - steps (int)
            - Number of gradient updates (samples to train on).
        - savepath (str)
            - Path where the model will be saved/loaded from.
            - e.g. hello/world/save_model.pth
        - resuming (bool)
            - Whether to resume training from a saved model or to start from scratch.
    """
    # Initialize model and send to device.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    model = models.BabyResNet().to(device)
    if resuming:
        model.load_state_dict(torch.load(savepath))
    # Declare optimizer, preprocessor, and list to record errors.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    preproc = ImageDataPipeline()
    errors = []
    # Begin training.
    for step, path, image, label in tqdm(preproc.preprocess_classes(steps, train_dir, label_dict),
                                         desc='Progress', total=steps, ncols=99, unit='image'):
        optimizer.zero_grad()
        image, label = torch.tensor(image).to(device), torch.tensor(label).to(device)
        output = model(image)
        error = torch.sqrt(torch.nn.functional.mse_loss(output, label))
        errors.append(error)
        error.backward()
        optimizer.step()
    save(model, savepath)
    print('\a')
    # Plot errors.
    plt.plot(np.array(list(range(1, steps + 1))), np.array(errors))
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.show()
