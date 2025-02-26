"""Abstract class for lifting hyper graphs to simplicial complexes."""

import torch
from toponetx.classes import SimplicialComplex

from topobenchmark.data.utils.utils import get_complex_connectivity
from topobenchmark.transforms.liftings.liftings import HypergraphLifting


class Hypergraph2SimplicialLifting(HypergraphLifting):
    r"""Abstract class for lifting hyper graphs to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "hypergraph2simplicial"
        self.signed = kwargs.get("signed", False)

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex
    ) -> dict:
        r"""Return the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            simplicial_complex,
            self.complex_dim,
            neighborhoods=self.neighborhoods,
            signed=self.signed,
        )
        # lifted_topology["x_0"] = torch.stack(
        #     list(
        #         simplicial_complex.get_simplex_attributes(
        #             "features", 0
        #         ).values()
        #     )
        # )
        return lifted_topology
