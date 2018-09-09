"""Provides interface for training the model."""

import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from . import models
from .preprocessing import ImageDataPipeline


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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    model = models.BabyResNet().to(device)
    if resuming:
        model.load_state_dict(torch.load(savepath))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    preproc = ImageDataPipeline()
    for step, path, image, label in tqdm(preproc.preprocess_classes(steps, train_dir, label_dict),
                                         desc='Progress', total=steps, ncols=99, unit='image'):
        optimizer.zero_grad()
        image, label = torch.tensor(image).to(device), torch.tensor(label).to(device)
        output = model(image)
        objective = torch.sqrt(torch.nn.functional.mse_loss(output, label))
        objective.backward()
        optimizer.step()
        # print('Step: {}/{} | Image: {} | Objective: {}'.format(step, steps, img_path, objective))
    savedir = Path(Path(savepath).parent)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    torch.save(model.state_dict(), savepath)
    print('\a')
