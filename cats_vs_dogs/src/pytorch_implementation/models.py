"""Provides definitions for the model's architecture."""

import torch.nn as nn
import torch.nn.functional as nnf


class BabyResNet(nn.Module):

    """Definition for model: Baby Resnet"""

    def __init__(self):
        """
        Defining layers with trainable variables.
        """
        super().__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        # Conv Block 2
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        # Conv Block 3
        self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv9 = nn.Conv2d(128, 128, 3, padding=1)
        # Block 4
        self.conv10 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv11 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)
        # Dense Block
        self.dense1 = nn.Linear(256, 2)

    def forward(self,
                input_):
        """
        Forward method for torch.nn.Module.

        Parameters:
            - input_ (tensor)
                - Shape must be [1, 3, 128, 128] as in NCHW.
        Returns:
            - The output of the model.
            - A [1, 2] shape tensor.
        """
        # Conv Block 1
        skip = self.conv1(input_)
        x = nnf.relu(skip)
        x = self.conv2(x)
        x = nnf.relu(x)
        x = self.conv3(x)
        x += skip
        x = nnf.relu(x)
        x = nnf.max_pool2d(x, 2)
        # Conv Block 2
        skip = self.conv4(x)
        x = nnf.relu(skip)
        x = self.conv5(x)
        x = nnf.relu(x)
        x = self.conv6(x)
        x += skip
        x = nnf.relu(x)
        x = nnf.max_pool2d(x, 2)
        # Conv Block 3
        skip = self.conv7(x)
        x = nnf.relu(skip)
        x = self.conv8(x)
        x = nnf.relu(x)
        x = self.conv9(x)
        x += skip
        x = nnf.relu(x)
        x = nnf.max_pool2d(x, 2)
        # Conv Block 4
        skip = self.conv10(x)
        x = nnf.relu(skip)
        x = self.conv11(x)
        x = nnf.relu(x)
        x = self.conv12(x)
        x += skip
        x = nnf.relu(x)
        x = nnf.max_pool2d(x, 2)
        # Dense Block
        x = nnf.avg_pool2d(x, 8)
        x = x.reshape(1, 256)
        x = self.dense1(x)
        return x
