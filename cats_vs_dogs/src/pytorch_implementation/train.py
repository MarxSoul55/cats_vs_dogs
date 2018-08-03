"""Provides interface for training the model."""

import torch

import constants as c
import models
from preprocessing import ImageDataPipeline


def main(steps,
         savedir,
         resuming=True):
    """
    Trains the model and saves the result.

    Parameters:
        - steps (int)
            - Number of gradient updates (samples to train on).
        - savedir (str)
            - Directory to save the model to.
        - resuming (bool)
            - Whether to resume training from a saved model or to start from scratch.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.BabyResNet().to(device)
    base_objective = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    preprocessor = ImageDataPipeline()
    for step, im_path, im_array, im_label in preprocessor.preprocess_classes(steps,
                                                                             c.TRAIN_DIR,
                                                                             c.ENCODING):
        im_array, im_label = (torch.tensor(im_array, dtype=torch.float32).to(device),
                              torch.tensor(im_label, dtype=torch.float32).to(device))
        optimizer.zero_grad()
        output = model(im_array)
        objective = torch.sqrt(base_objective(output, im_label))
        objective.backward()
        optimizer.step()
        print('Step: {} | Image: {} | Objective: {}'.format(step, im_path, objective))
