"""Provides interface for training the model."""

import torch

from . import models
from .preprocessing import ImageDataPipeline


def main(train_dir,
         steps,
         savepath,
         resuming=True):
    """
    Trains the model and saves the result.

    Parameters:
        - train_dir (str)
            - Path to the directory of classes.
            - e.g. 'data/train', where 'train' holds subdirs with images in them.
        - steps (int)
            - Number of gradient updates (samples to train on).
        - savepath (str)
            - Path where the model will be saved/loaded from.
            - e.g. hello/world/save_model.pth
        - resuming (bool)
            - Whether to resume training from a saved model or to start from scratch.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.BabyResNet().to(device)
    if resuming:
        model.load_state_dict(torch.load(savepath))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    preprocessor = ImageDataPipeline()
    for step, img_path, img_tensor, img_label in preprocessor.preprocess_classes(steps,
                                                                                 train_dir):
        img_tensor, img_label = (torch.tensor(img_tensor, dtype=torch.float32).to(device),
                                 torch.tensor(img_label, dtype=torch.float32).to(device))
        optimizer.zero_grad()
        output = model(img_tensor)
        objective = torch.sqrt(torch.nn.functional.mse_loss(output, img_label))
        objective.backward()
        optimizer.step()
        print('Step: {} | Image: {} | Objective: {}'.format(step, img_path, objective))
    torch.save(model.state_dict(), savepath)
    print('\a')
