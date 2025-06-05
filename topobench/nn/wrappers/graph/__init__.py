"""Wrappers for graph models."""

from .gatv4_wrapper import GATv4Wrapper
from .gnn_wrapper import GNNWrapper
from .graph_mlp_wrapper import GraphMLPWrapper

# Export all wrappers
__all__ = [
    "GATv4Wrapper",
    "GNNWrapper",
    "GraphMLPWrapper",
]
