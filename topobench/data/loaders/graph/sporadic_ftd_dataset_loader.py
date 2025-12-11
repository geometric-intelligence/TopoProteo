"""Loader for Sporadic FTD Dataset."""
import os
from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets.sporadic_ftd_dataset import SporadicFTDDataset
from topobench.data.loaders.base import AbstractLoader
import numpy as np


class SporadicFTDDatasetLoader(AbstractLoader):
    """Load Sporadic FTD Dataset (OOD test set).

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for sporadic FTD data
            - ftd_root: Root directory for FTD dataset (to load scalers/adjacency)
            - data_name: Name of the dataset
            - All other FTD config parameters (for matching experiment IDs)
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.datasets = []

    def load_dataset(self) -> SporadicFTDDataset:
        """Load the Sporadic FTD dataset.

        Returns
        -------
        SporadicFTDDataset
            The loaded Sporadic FTD dataset (combined, with split_idx set).

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        self._load_splits()
        # Prepare split_idx so all data is marked as test set
        # This allows masks to be assigned correctly via assign_train_val_test_mask_to_graphs
        split_idx = self._prepare_split_idx()
        # Combine splits (even though there's only one split, we combine for consistency)
        combined_dataset = self._combine_splits()
        combined_dataset.split_idx = split_idx
        self.config_tag = self.datasets[0].config_tag
        return combined_dataset
        
    def get_data_dir(self):
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, "processed", self.config_tag)

    def get_splits(self) -> list:
        """Get the dataset splits.

        Returns
        -------
        list
            A list containing the test dataset (sporadic is only test set).
        """
        self._load_splits()
        return self.datasets

    def _load_splits(self) -> None:
        """Load the dataset split (only test for sporadic FTD)."""
        # Sporadic FTD is only used as test set
        split = "test"
        print(f"Loading Sporadic FTD dataset split: {split}")
        ftd_root = self.parameters.get("ftd_root")
        if ftd_root is None:
            raise ValueError(
                "ftd_root must be specified in config to load FTD scalers and adjacency matrix"
            )
        
        dataset = SporadicFTDDataset(
            root=str(self.root_data_dir),
            config=self.parameters,
            ftd_root=str(ftd_root),
            split=split,
        )
        self.datasets.append(dataset)
        self.config_tag = dataset.config_tag

    def _prepare_split_idx(self) -> dict[str, np.ndarray]:
        """Prepare the split indices for the dataset.
        
        Since sporadic FTD is only used as a test set, all indices are marked as test.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping split names to index arrays.
        """
        # All data is test data for sporadic FTD
        split_idx = {
            "train": np.array([], dtype=np.int64),  # Empty train set
            "valid": np.array([], dtype=np.int64),  # Empty validation set
            "test": np.arange(len(self.datasets[0])),  # All data is test
        }
        return split_idx

    def _combine_splits(self):
        """Combine the dataset splits into a single dataset.

        For sporadic FTD, there's only one split (test), but we combine it
        for consistency with the FTD loader pattern.

        Returns
        -------
        Dataset
            The combined dataset (which is just the test dataset).
        """
        return self.datasets[0]

