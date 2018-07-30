"""Provides definitions for the model's architecture."""

import torch


class BabyResnet(torch.nn.Module):

    """Definition for model: Baby Resnet"""

    def __init__(self):
        super().__init__()

    def forward(self,
                input_):
        """
        Forward method for torch.nn.Module.

        Parameters:
            - input_ (tensor)
                - The input as a NCHW formatted tensor.
        Returns:
            - The output of the model.
            - A [1, 2] shape tensor.
        """
        # Conv Block 1
        skip = torch.nn.Conv2d(3, 32, 3, padding=1)(input_)
        x = torch.nn.ReLU()(skip)
        x = torch.nn.Conv2d(32, 32, 3, padding=1)(x)
        x = torch.nn.ReLU()(x)
        x = torch.nn.Conv2d(32, 32, 3, padding=1)(x)
        x += skip
        x = torch.nn.ReLU()(x)
        x = torch.nn.MaxPool2d(2)(x)
        # Conv Block 2
        skip = torch.nn.Conv2d(32, 64, 3, padding=1)(x)
        x = torch.nn.ReLU()(skip)
        x = torch.nn.Conv2d(64, 64, 3, padding=1)(x)
        x = torch.nn.ReLU()(x)
        x = torch.nn.Conv2d(64, 64, 3, padding=1)(x)
        x += skip
        x = torch.nn.ReLU()(x)
        x = torch.nn.MaxPool2d(2)(x)
        # Conv Block 3
        skip = torch.nn.Conv2d(64, 128, 3, padding=1)(x)
        x = torch.nn.ReLU()(skip)
        x = torch.nn.Conv2d(128, 128, 3, padding=1)(x)
        x = torch.nn.ReLU()(x)
        x = torch.nn.Conv2d(128, 128, 3, padding=1)(x)
        x += skip
        x = torch.nn.ReLU()(x)
        x = torch.nn.MaxPool2d(2)(x)
        # Conv Block 4
        skip = torch.nn.Conv2d(128, 256, 3, padding=1)(x)
        x = torch.nn.ReLU()(skip)
        x = torch.nn.Conv2d(256, 256, 3, padding=1)(x)
        x = torch.nn.ReLU()(x)
        x = torch.nn.Conv2d(256, 256, 3, padding=1)(x)
        x += skip
        x = torch.nn.ReLU()(x)
        x = torch.nn.MaxPool2d(2)(x)
        # Dense Block
        x = torch.nn.AvgPool2d(x.shape[2])
        x = x.reshape(1, -1)
        x = torch.nn.Linear(x.shape[1], 2)(x)
        return x
