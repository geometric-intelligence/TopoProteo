"""Class to apply BaseEncoder to the features of higher order structures."""

import torch
import torch_geometric

from topobench.nn.encoders.base import AbstractFeatureEncoder


class IdentityEncoder(AbstractFeatureEncoder):
    r"""Identity encoder class.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass.

        The method applies the BaseEncoders to the features of the selected_dimensions.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x_{i} features for each i in the selected_dimensions.

        Returns
        -------
        torch_geometric.data.Data
            Output data object with updated x_{i} features.
        """
        if not hasattr(data, "x_0"):
            data.x_0 = data.x
        return data

