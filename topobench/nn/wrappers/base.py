"""Abstract class that provides an interface to handle the network output."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractWrapper(ABC, torch.nn.Module):
    r"""Abstract class that provides an interface to handle the network output.

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone model.
    features_to_keep : list, optional
        List of features to keep in the output. If None, no features are kept.
    **kwargs : dict
        Additional arguments for the class. It should contain the following keys:
        - out_channels (int): Number of output channels.
        - num_cell_dimensions (int): Number of cell dimensions.
    """

    def __init__(self, backbone, features_to_keep=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        out_channels = kwargs["out_channels"]
        self.dimensions = range(kwargs.get("num_cell_dimensions",1))
        self.residual_connections = kwargs.get("residual_connections", True)
        self.features_to_keep = features_to_keep if features_to_keep else []

        for i in self.dimensions:
            setattr(
                self,
                f"ln_{i}",
                nn.LayerNorm(out_channels),
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(backbone={self.backbone}, out_channels={self.backbone.out_channels}, dimensions={self.dimensions}, residual_connections={self.residual_connections})"

    def __call__(self, batch):
        r"""Forward pass for the model.

        This method calls the forward method and adds the residual connection.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        model_out = self.forward(batch)
        model_out = (
            self.residual_connection(model_out=model_out, batch=batch)
            if self.residual_connections
            else model_out
        )
        # Add the features to keep to the model output, if any
        for feature in self.features_to_keep:
            model_out[feature] = batch[feature]
        return model_out

    def residual_connection(self, model_out, batch):
        r"""Residual connection for the model.

        This method sums, for the embeddings of the cells of any rank, the
        output of the model with the input embeddings and applies layer
        normalization.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        for i in self.dimensions:
            if (
                (f"x_{i}" in batch)
                and hasattr(self, f"ln_{i}")
                and (f"x_{i}" in model_out)
            ):
                residual = model_out[f"x_{i}"] + batch[f"x_{i}"]
                model_out[f"x_{i}"] = getattr(self, f"ln_{i}")(residual)
        return model_out

    @abstractmethod
    def forward(self, batch):
        r"""Forward pass for the model.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.
        """
