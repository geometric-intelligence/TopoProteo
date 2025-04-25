"""Loaders for US County Demos dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets import FTDDataset
from topobench.data.loaders.base import AbstractLoader
import numpy as np


class FTDDatasetLoader(AbstractLoader):
    """Load FTD Dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - year: Year of the dataset (if applicable)
            - task_variable: Task variable for the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.datasets = []

    def load_dataset(self) -> FTDDataset:
        """Load the FTD dataset.

        Returns
        -------
        FTDDataset
            The loaded US County Demos dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        self._load_splits()
        split_idx = self._prepare_split_idx()
        combined_dataset = self._combine_splits()
        combined_dataset.split_idx = split_idx
        return combined_dataset
        # dataset = self._initialize_dataset()
        # self.data_dir = self._redefine_data_dir(dataset)
        # return dataset

    # def _initialize_dataset(self) -> FTDDataset:
    #     """Initialize the US County Demos dataset.

    #     Returns
    #     -------
    #     FTDDataset
    #         The initialized dataset instance.
    #     """
    #     train_dataset = FTDDataset(
    #             root=str(self.root_data_dir),
    #             config=self.parameters,
    #             split="train",
    #         )
    #     val_dataset = FTDDataset(
    #             root=str(self.root_data_dir),
    #             config=self.parameters,
    #             split="val",
    #         )
    #     return (train_dataset, val_dataset)

    def _load_splits(self) -> None:
        """Load the dataset splits for the specified dataset."""
        for split in ["train", "val", "val"]:
            self.datasets.append(
                FTDDataset(
                    root=str(self.root_data_dir),
                    config=self.parameters,
                    split=split,
                )
            )

    def _prepare_split_idx(self) -> dict[str, np.ndarray]:
        """Prepare the split indices for the dataset.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping split names to index arrays.
        """
        split_idx = {"train": np.arange(len(self.datasets[0]))}
        split_idx["valid"] = np.arange(
            len(self.datasets[0]),
            len(self.datasets[0]) + len(self.datasets[1]),
        )
        split_idx["test"] = np.arange(
            len(self.datasets[0]) + len(self.datasets[1]),
            len(self.datasets[0])
            + len(self.datasets[1])
            + len(self.datasets[2]),
        )
        return split_idx

    def _combine_splits(self):
        """Combine the dataset splits into a single dataset.

        Returns
        -------
        Dataset
            The combined dataset containing all splits.
        """
        return self.datasets[0] + self.datasets[1] + self.datasets[2]

    def _redefine_data_dir(self, dataset: FTDDataset) -> Path:
        """Redefine the data directory based on the chosen variable.

        Parameters
        ----------
        dataset : FTDDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root
