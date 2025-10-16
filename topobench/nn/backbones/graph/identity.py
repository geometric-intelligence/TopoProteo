"""
This module contains the implementation of identity GNNs.
"""

import torch


class Identity(torch.nn.Module):
    """Identity."""

    def __init__(
        self,
        **kwargs,  # Additional keyword arguments
    ):
        super().__init__()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        return x
