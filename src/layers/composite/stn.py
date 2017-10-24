"""
Provides interface to a spatial-transformer network.
See https://arxiv.org/pdf/1506.02025.pdf for details.
"""

from core.convolutional import convolution_2d
from core.misc import dense
from core.activations import elu


def stn(input_):
    """
    Uses a small, convolutional neural-network to implement a spatial-transformer network.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
        # TODO
    # Returns
        # TODO
    """
    # Localization network.
        # TODO
    # Grid generator.
        # TODO
    # Sampler.
        # TODO
