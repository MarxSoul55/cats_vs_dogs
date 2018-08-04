"""Provides interface for training the model."""

import torch

import models
from preprocessing import ImageDataPipeline


def main(train_dir,
         encoding,
         steps,
         savepath,
         resuming=True):
    """
    Trains the model and saves the result.

    Parameters:
        - train_dir, encoding
            - Parameters for the preprocessor.
            - See `preprocessing.ImageDataPipeline.preprocess_classes` for details.
        - steps (int)
            - Number of gradient updates (samples to train on).
        - savepath (str)
            - Path where the model will be saved/loaded from.
            - e.g. hello/world/save_model.pth
        - resuming (bool)
            - Whether to resume training from a saved model or to start from scratch.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if resuming:
        model = torch.load(savepath).to(device)
    else:
        model = models.BabyResNet().to(device)
    base_objective = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    preprocessor = ImageDataPipeline()
    for step, im_path, im_array, im_label in preprocessor.preprocess_classes(steps,
                                                                             train_dir,
                                                                             encoding):
        im_array, im_label = (torch.tensor(im_array, dtype=torch.float32).to(device),
                              torch.tensor(im_label, dtype=torch.float32).to(device))
        optimizer.zero_grad()
        output = model(im_array)
        objective = torch.sqrt(base_objective(output, im_label))
        objective.backward()
        optimizer.step()
        print('Step: {} | Image: {} | Objective: {}'.format(step, im_path, objective))
    torch.save(model, savepath)
