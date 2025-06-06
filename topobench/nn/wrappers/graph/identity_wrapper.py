"""Wrapper for the GNN models."""

from topobench.nn.wrappers.base import AbstractWrapper


class IdentityWrapper(AbstractWrapper):
    r"""Identity Wrapper."""

    def forward(self, batch):
        r"""Forward pass for the Identity wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = batch.x_0

        return model_out
