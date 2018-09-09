"""Provides interface for training the model."""

import os
from pathlib import Path

import torch

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
    # Initialize device, model, and optimizer.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.BabyResNet().to(device)
    if resuming:
        model.load_state_dict(torch.load(savepath))
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)
    # Initialize preprocessor and begin training the model.
    preprocessor = ImageDataPipeline()
    for step, img_path, img_tensor, img_label in preprocessor.preprocess_classes(steps,
                                                                                 train_dir,
                                                                                 label_dict):
        img_tensor, img_label = (torch.tensor(img_tensor, dtype=torch.float32).to(device),
                                 torch.tensor(img_label, dtype=torch.float32).to(device))
        optimizer.zero_grad()
        output = model(img_tensor)
        objective = torch.sqrt(torch.nn.functional.mse_loss(output, img_label))
        objective.backward()
        optimizer.step()
        print('Step: {}/{} | Image: {} | Objective: {}'.format(step, steps, img_path, objective))
    # Create savedir if nonexistent, and save the model.
    savedir = Path(Path(savepath).parent)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    torch.save(model.state_dict(), savepath)
    # Play a noise to signify end of training.
    print('\a')
