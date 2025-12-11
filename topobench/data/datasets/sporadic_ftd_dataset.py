import os
import numpy as np
import pandas as pd
import torch
import joblib
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.datasets.ftd_dataset import (
    LABEL_DIM_MAP,
    SEXES,
    MODALITIES,
    Y_VALS_TO_NORMALIZE,
    BINARY_Y_VALS_MAP,
    MULTICLASS_Y_VALS_MAP,
    HAS_MODALITY_COL,
    MODALITY_COL_END,
    Y_VAL_COL_MAP,
    sex_col,
    age_col,
    did_col,
    gene_col,
    remove_erroneous_columns_and_two_pass_error_proteins,
    load_scalers,
    load_protein_columns,
    log_transform,
    plot_histogram,
)


class SporadicFTDDataset(InMemoryDataset):
    """Sporadic FTD dataset used as OOD test set.
    
    This dataset loads an external CSV file and applies the same preprocessing
    (scaling, column ordering) as the FTD dataset. It uses scalers and adjacency
    matrix from the corresponding FTD dataset.
    
    Key differences from FTDDataset:
    - No mutation column (sporadic FTD patients don't have known mutations)
    - Loads scalers and protein column order from FTD processed directory
    - Uses the same adjacency matrix from FTD
    """

    def __init__(self, root, config, ftd_root, split="test"):
        """
        Parameters
        ----------
        root : str
            Root directory for this dataset (where processed files will be saved)
        config : object
            Configuration object with same structure as FTD config
        ftd_root : str
            Root directory of the FTD dataset (to load scalers and adjacency matrix)
        split : str
            Split name (default: "test" since this is always used as test set)
        """
        self.name = "SporadicFTD"
        self.root = root
        self.ftd_root = ftd_root
        self.split = split
        assert self.split == "test", "SporadicFTD dataset is only used as test set"
        
        assert config.sex in SEXES
        assert config.modality in MODALITIES
        assert config.y_val in LABEL_DIM_MAP
        
        # If adj_metric is set to pointcloud, the adjacency matrix is the identity
        config.adj_thresh = 1.0 if config.adj_metric == "pointcloud" else config.adj_thresh

        self.config = config
        self.adj_metric = config.adj_metric
        self.adj_str = f"adj_thresh_{config.adj_thresh}"
        self.y_val_str = f"y_val_{config.y_val}"
        self.num_nodes_str = f"num_nodes_{config.num_nodes}"
        self.modality_str = f"{config.modality}"
        self.sex_str = f"sex_{','.join(config.sex)}"
        
        # Use FTD experiment ID to load scalers and adjacency matrix
        # Get FTD mutation list from config (only used for matching experiment ID, not for filtering)
        ftd_mutation = config.get("ftd_mutation", config.get("mutation", []))
        if not ftd_mutation:
            raise ValueError(
                "ftd_mutation must be specified in config to match FTD experiment ID. "
                "This is only used to identify which FTD scalers/adjacency matrix to load, "
                "not for filtering sporadic data (which has no mutations)."
            )
        self.ftd_experiment_id = (
            f"FTD_{self.y_val_str}_{self.adj_metric}_{self.adj_str}_{self.num_nodes_str}_"
            f"mutation_{','.join(ftd_mutation)}_{self.modality_str}_{self.sex_str}"
        )
        
        # Compute experiment_id and config_tag before super().__init__()
        # so they're available when processed_file_names is called
        self.experiment_id = (
            f"{self.name}_{self.y_val_str}_{self.adj_metric}_{self.adj_str}_"
            f"{self.num_nodes_str}_{self.modality_str}_{self.sex_str}"
        )
        
        if config.kfold:
            self.config_tag = (
                f"{self.experiment_id}_random_state_{config.random_state}_"
                f"{config.num_folds}fold_{config.fold}_two_pass_{config.two_pass}"
            )
        else:
            self.config_tag = (
                f"{self.experiment_id}_random_state_{config.random_state}_"
                f"two_pass_{config.two_pass}"
            )
        
        # Path to FTD adjacency matrix - must be set BEFORE super().__init__()
        # because process() will be called and needs adj_path
        ftd_processed_dir = os.path.join(ftd_root, "processed")
        ftd_mutation = config.get("ftd_mutation", config.get("mutation", []))
        if config.kfold:
            self.adj_path = os.path.join(
                ftd_processed_dir,
                f"adjacency_num_nodes_{config.num_nodes}_{self.adj_metric}_"
                f"mutation_{ftd_mutation}_{config.modality}_sex_{config.sex}_"
                f"random_state_{config.random_state}_{config.num_folds}fold_{config.fold}_"
                f"two_pass_{config.two_pass}.csv",
            )
        else:
            self.adj_path = os.path.join(
                ftd_processed_dir,
                f"adjacency_num_nodes_{config.num_nodes}_{self.adj_metric}_"
                f"mutation_{ftd_mutation}_{config.modality}_sex_{config.sex}_"
                f"random_state_{config.random_state}_two_pass_{config.two_pass}.csv",
            )
        
        super(SporadicFTDDataset, self).__init__(
            root, transform=None, pre_transform=None
        )
        
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = LABEL_DIM_MAP[self.config.y_val]
        
        # Construct path using config_tag (already computed before super().__init__())
        path = os.path.join(
            self.processed_dir,
            f"{self.config_tag}_{self.split}.pt",
        )
        
        print("Loading data from:", path)
        self.load(path)

    @property
    def raw_file_names(self):
        """Files that must be present in order to skip downloading."""
        return [self.config.raw_file_name]

    @property
    def processed_file_names(self):
        """Files that must be present in order to skip processing."""
        # experiment_id and config_tag are computed in __init__ before super().__init__()
        files = [
            f"{self.config_tag}_test.pt",
        ]
        print("Processed file names:", files)
        return files

    def create_graph_data(
        self,
        feature,
        label,
        adj_matrix,
        sex,
        age,
    ):
        """Create Data object for each graph.
        
        Note: mutation is not included since sporadic FTD doesn't have mutations.
        """
        x = feature  # protein concentrations: what is on the nodes
        adj_tensor = torch.tensor(adj_matrix)
        # Find the indices where the matrix has non-zero elements
        pairs_indices = torch.nonzero(adj_tensor, as_tuple=False)
        # Extract the pairs of connected nodes
        edge_index = pairs_indices.t().contiguous()
        sex = sex.unsqueeze(1)
        age = age.unsqueeze(1)
        # Create dummy mutation tensor (all zeros) since sporadic has no mutations
        mutation = torch.zeros_like(sex)
        return Data(
            x=x,
            edge_index=edge_index,
            y=label,
            sex=sex,
            mutation=mutation,
            age=age,
        )

    def process(self):
        """Read data into Data list, applying FTD scalers and preprocessing."""
        # Load the external CSV data
        (
            features,
            labels,
            protein_cols,
            filtered_sex_col,
            filtered_age_col,
            filtered_did_col,
        ) = self.load_csv_data_pre_pt_files(self.config)

        # Convert sex to categorical labels
        sex_labels = np.array(filtered_sex_col.astype("category").cat.codes)
        
        # Process the data using FTD scalers
        (
            test_features,
            test_labels,
            test_sex,
            test_age,
            adj_matrix,
        ) = self.load_csv_data_with_ftd_scalers(
            self.config,
            features,
            labels,
            protein_cols,
            sex_labels,
            filtered_age_col.values,
        )

        test_data_list = []
        # Iterate through test data and use the FTD adjacency matrix
        for feature, label, sex, age in zip(
            test_features, test_labels, test_sex, test_age
        ):
            data = self.create_graph_data(
                feature, label, adj_matrix, sex, age
            )
            test_data_list.append(data)

        # Save the test data list
        test_path = f"{self.processed_paths[0]}"
        self.save(test_data_list, test_path)

    def load_y_vals(self, filtered_data):
        """Find the y_val values based on the config."""
        y_vals = filtered_data[Y_VAL_COL_MAP[self.config.y_val]]
        y_vals_mask = ~y_vals.isna()
        y_vals = y_vals[y_vals_mask]

        if self.config.y_val in BINARY_Y_VALS_MAP:
            y_vals = self.load_binary_y_values(y_vals)
        elif self.config.y_val in MULTICLASS_Y_VALS_MAP:
            y_vals = self.load_multiclass_y_values(y_vals)

        return y_vals, y_vals_mask

    def load_binary_y_values(self, y_vals):
        """Load the binary y_val values using dictionary that maps values to keys."""
        mapping_dict = BINARY_Y_VALS_MAP[self.config.y_val]
        mapped_values = [mapping_dict[value] for value in y_vals]
        return mapped_values

    def load_multiclass_y_values(self, y_vals):
        """Load multiclass y_values and encode as index targets for focal loss"""
        mapping_dict = MULTICLASS_Y_VALS_MAP[self.config.y_val]
        mapped_values = [mapping_dict[value] for value in y_vals]
        return mapped_values

    def load_csv_data_pre_pt_files(self, config):
        """Load the CSV data features and labels. Filter by sex only (all sporadic data is CSF)."""
        csv_path = self.raw_paths[0]
        print("Loading sporadic FTD data from:", csv_path)
        csv_data = pd.read_csv(csv_path)

        # Remove erroneous columns (same as FTD)
        csv_data = remove_erroneous_columns_and_two_pass_error_proteins(
            config, csv_data, self.raw_dir
        )

        # Filter by sex only (no modality filtering - all sporadic data is CSF)
        condition_sex = csv_data[sex_col].isin(self.config.sex)
        filtered_data = csv_data[condition_sex]
        
        print(
            f"Number of patients with sex in {self.config.sex}:",
            condition_sex.sum(),
        )
        print(
            "Total number of patients after filtering:",
            len(filtered_data),
        )

        # Extract the y_val values
        y_vals, y_val_mask = self.load_y_vals(filtered_data)
        filtered_data = filtered_data[y_val_mask]  # Remove rows where y_val is NaN
        print("final dims of filtered data:", filtered_data.shape)

        # Extract column labels
        filtered_sex_col = filtered_data[sex_col]
        filtered_age_col = filtered_data[age_col]
        filtered_did_col = filtered_data[did_col]

        # Get protein columns - will be reordered later to match FTD
        protein_cols = [
            col
            for col in filtered_data.columns
            if col.endswith(MODALITY_COL_END[self.config.modality])
        ]
        print("Number of proteins in sporadic dataset:", len(protein_cols))
        
        # Extract features (will be reordered later)
        features = np.array(filtered_data[protein_cols], dtype=np.float32)
        labels = np.array(y_vals, dtype=np.float32)

        return (
            features,
            labels,
            protein_cols,  # Return column names for proper reordering
            filtered_sex_col,
            filtered_age_col,
            filtered_did_col,
        )

    def load_csv_data_with_ftd_scalers(
        self,
        config,
        features,
        labels,
        protein_cols,
        sex_labels,
        age_values,
    ):
        """Load and scale data using FTD scalers."""
        # Load FTD processed directory
        ftd_processed_dir = os.path.join(self.ftd_root, "processed")
        
        # Load scalers from FTD
        scalers = load_scalers(config, self.ftd_experiment_id, ftd_processed_dir)
        feature_scaler = scalers['feature_scaler']
        age_scaler = scalers['age_scaler']
        sex_scaler = scalers['sex_scaler']
        
        # Load protein column order from FTD
        ftd_protein_columns = load_protein_columns(
            config, self.ftd_experiment_id, ftd_processed_dir
        )
        
        print(f"FTD has {len(ftd_protein_columns)} protein columns")
        print(f"Sporadic dataset has {len(protein_cols)} protein columns")
        
        # Reorder features to match FTD protein column order exactly
        # Create a DataFrame with sporadic protein columns as column names
        features_df = pd.DataFrame(features, columns=protein_cols)
        
        # Initialize reordered features array with zeros (for missing columns)
        features_reordered = np.zeros((features.shape[0], len(ftd_protein_columns)), dtype=np.float32)
        
        # Map sporadic columns to FTD column order
        for idx, ftd_col in enumerate(ftd_protein_columns):
            if ftd_col in features_df.columns:
                features_reordered[:, idx] = features_df[ftd_col].values
            else:
                # Column missing in sporadic dataset - fill with 0
                print(f"Warning: Column {ftd_col} not found in sporadic dataset, filling with 0")
                features_reordered[:, idx] = 0.0
        
        # Check for columns in sporadic that are not in FTD
        missing_in_ftd = set(protein_cols) - set(ftd_protein_columns)
        if missing_in_ftd:
            print(f"Warning: {len(missing_in_ftd)} columns in sporadic dataset not in FTD (will be ignored)")
        
        # Normalize labels if needed
        if config.y_val in Y_VALS_TO_NORMALIZE:
            # Load mean and std from FTD
            import json
            if config.kfold:
                stats_file = os.path.join(
                    ftd_processed_dir,
                    f"{self.ftd_experiment_id}_train_random_state_{config.random_state}_{config.num_folds}fold_{config.fold}.json"
                )
            else:
                stats_file = os.path.join(
                    ftd_processed_dir,
                    f"{self.ftd_experiment_id}_train_random_state_{config.random_state}.json"
                )
            
            with open(stats_file, "r") as f:
                content = f.read()
                mean = float(content.split("mean: ")[1].split("\n")[0])
                std = float(content.split("std: ")[1].split("\n")[0])
            
            # Apply normalization using FTD mean/std
            labels_norm = (labels - mean) / std
            labels_norm = labels_norm.astype(np.float32)
        else:
            labels_norm = labels.astype(np.float32) if isinstance(labels, np.ndarray) else labels

        # Scale features using FTD feature scaler
        features_scaled = feature_scaler.transform(features_reordered).astype(np.float32)
        
        # Scale age using FTD age scaler
        age_scaled = age_scaler.transform(age_values.reshape(-1, 1)).astype(np.float32)
        
        # Scale sex using FTD sex scaler
        sex_scaled = sex_scaler.transform(sex_labels.reshape(-1, 1)).astype(np.float32)

        # Convert to tensors
        test_features = torch.FloatTensor(
            features_scaled.reshape(-1, features_scaled.shape[1], 1)
        )
        test_labels_norm = torch.FloatTensor(labels_norm)
        test_sex = torch.FloatTensor(sex_scaled)
        test_age = torch.FloatTensor(age_scaled)

        print("Test features and labels:", test_features.shape, test_labels_norm.shape)
        print("Test sex and age labels shape:", test_sex.shape, test_age.shape)

        # Load adjacency matrix from FTD
        adj_matrix = self.get_adjacency_matrix(
            self.adj_path, config.adj_thresh, config
        )

        return (
            test_features,
            test_labels_norm,
            test_sex,
            test_age,
            adj_matrix,
        )

    def get_adjacency_matrix(self, path, adj_thresh, config):
        """Load and threshold an adjacency matrix from FTD."""
        print(f"Loading adjacency matrix from: {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Adjacency matrix not found at: {path}. "
                "Please process FTD dataset first."
            )
        adj_matrix = np.array(pd.read_csv(path, header=None)).astype(float)
        adj_matrix = torch.FloatTensor(
            np.where(adj_matrix >= adj_thresh, 1, 0)
        )  # Thresholding
        print("Adjacency matrix shape:", adj_matrix.shape)
        expected_shape = (config.num_nodes, config.num_nodes)
        assert adj_matrix.shape == expected_shape, (
            f"Unexpected shape: {adj_matrix.shape}. Expected shape: {expected_shape}"
        )
        print("Number of edges:", adj_matrix.sum())
        return adj_matrix

