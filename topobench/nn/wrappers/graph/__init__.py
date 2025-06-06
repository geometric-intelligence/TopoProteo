"""Wrappers for graph models."""

from .gatv4_wrapper import GATv4Wrapper
from .gnn_wrapper import GNNWrapper
from .graph_mlp_wrapper import GraphMLPWrapper
from .identity_wrapper import IdentityWrapper

# Export all wrappers
__all__ = [
    "GATv4Wrapper",
    "GNNWrapper",
    "GraphMLPWrapper",
    "IdentityWrapper",
]
