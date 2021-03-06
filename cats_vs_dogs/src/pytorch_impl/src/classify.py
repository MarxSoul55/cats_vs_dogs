"""Provides interface to classification with the model."""

import os

import numpy as np
import torch

from . import models
from .preprocessing import ImageDataPipeline


def predicted_label(prediction_tensor,
                    label_dict):
    """
    Generates the predicted label by comparing the tensor prediction to `label_dict`.

    Parameters:
        - prediction (np.ndarray)
            - The prediction as represented by the model's tensor output.
        - label_dict (dict, str --> np.ndarray)
            - See the parent function for details.
    Returns:
        - A string; the predicted label.
    """
    classes = list(label_dict.keys())
    labels = list(label_dict.values())
    differences = []
    for label in labels:
        l2_difference = np.sqrt(np.sum((label - prediction_tensor) ** 2))
        differences.append(l2_difference)
    index_of_smallest_difference = differences.index(min(differences))
    return classes[index_of_smallest_difference]


def main(src,
         savepath,
         label_dict):
    """
    Does one of 2 things:
    1. Given a path to an image file on disk (WITH A FILE-EXTENSION), classifies it.
    2. Given a path to a directory on disk, classifies all images found in it (excluding
       subdirectories and files with unsupported formats).

    Parameters:
        - src (str)
            - Can be a normal path to an image on disk.
            - Can also be a path to a directory.
        - savepath (str)
            - Path to a saved model on disk.
            - This model is used for the classification.
        - label_dict (dict, str --> np.ndarray)
            - Maps the name of the subdirectory (class) to a label.
                - ex: {'cats': np.array([[1, 0]]), 'dogs': np.array([[0, 1]])}
                - Each label must have the same shape!
            - The model's prediction is compared to each label.
                - Whichever label differs the least from the prediction is chosen to return.
    Returns:
        - If given path to an image file on disk, returns a string that is either 'cat' or 'dog'.
        - If given path to a directory, returns a dictionary {'filename': 'guessed animal'}
    """
    # Initialize device, model, and preprocessor.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.BabyResNet().to(device)
    model.load_state_dict(torch.load(savepath))
    model.eval()
    preprocessor = ImageDataPipeline()
    # If path leads to a directory, classify each image in it.
    if os.path.isdir(src):
        results = {}
        for img_path, img_tensor in preprocessor.preprocess_directory(src):
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32).to(device)
            output = model(img_tensor).cpu().detach().numpy()
            results[img_path] = predicted_label(output, label_dict)
        return results
    # Else, classify one image.
    img_tensor = torch.tensor(preprocessor.preprocess_image(src), dtype=torch.float32).to(device)
    output = model(img_tensor).cpu().detach().numpy()
    return predicted_label(output, label_dict)
