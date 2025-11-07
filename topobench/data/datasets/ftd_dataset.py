import os

import os.path as osp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import PyWGCNA
import torch
from scipy.stats import chi2_contingency, kendalltau, ks_2samp, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, InMemoryDataset
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Callable, Sequence

from topobench.data.utils.compute_adjacency_utils import (
    calculate_wgcna_matrix,
    # calculate_maximal_information_coefficient_matrix,
    calculate_spearman_correlation_matrix,
    calculate_distance_correlation_matrix,
    calculate_mutual_information_matrix,
)


LABEL_DIM_MAP = {
    "clinical_dementia_rating_global": 5,
    "clinical_dementia_rating_binary": 1,  # binary classification CDR=0 versus CDR>0
    "clinical_dementia_rating": 1,
    "carrier": 1,
    "disease_age": 1,
    "executive_function": 1,
    "memory": 1,
    "nfl": 1,
    "cog_z_score": 1,
    "global_cog_slope": 1,
}
SEXES = [["M"], ["F"], ["M", "F"], ["F", "M"]]
MODALITIES = ["plasma", "csf"]

Y_VALS_TO_NORMALIZE = [
    "nfl",
    "cog_z_score",
    "clinical_dementia_rating",
    "global_cog_slope",
]
CONTINOUS_Y_VALS = [
    "nfl",
    "disease_age",
    "executive_function",
    "memory",
    "clinical_dementia_rating",
    "cog_z_score",
    "global_cog_slope",
]
BINARY_Y_VALS_MAP = {
    "clinical_dementia_rating_binary": {0: 0, 0.5: 1, 1: 1, 2: 1, 3: 1},
    "carrier": {"CTL": 0, "Carrier": 1},
}
MULTICLASS_Y_VALS_MAP = {
    "clinical_dementia_rating_global": {0: 0, 0.5: 1, 1: 2, 2: 3, 3: 4}
}

HAS_MODALITY_COL = {
    "plasma": "HasPlasma?",
    "csf": "HasCSF?",
}
MODALITY_COL_END = {
    "plasma": "|PLASMA",
    "csf": "|CSF",
}
Y_VAL_COL_MAP = {
    "nfl": "NFL3_MEAN",
    "disease_age": "disease.age",
    "executive_function": "ef.unadj.slope",
    "memory": "mem.unadj.slope",
    "clinical_dementia_rating": "FTLDCDR_SB",
    "clinical_dementia_rating_global": "CDRGLOB",
    "clinical_dementia_rating_binary": "CDRGLOB",
    "carrier": "Carrier.Status",
    "cog_z_score": "GLOBALCOG.ZSCORE",
    "global_cog_slope": "global.ageadj.slope",
}

mutation_col = "Mutation"
sex_col = "SEX_AT_BIRTH"
age_col = "AGE_AT_VISIT"
did_col = "DID"
gene_col = "Gene.Dx"


class FTDDataset(InMemoryDataset):
    """This is dataset used in FTD.
    This is a graph regression task.

    **Rows:**
    - 0: Column Headers
    - 1 - 531 : Patient ID Number *(int)*

    **Columns:**
    - 0: DID *(int):* Patient ID
    - 1: Mutation *(string)*: CTL (Control), MAPT, C9orf72, GRN
    - 2: AGE_AT_VISIT *(int)*
    - 3: SEX_AT_BIRTH *(string)*: M, F
    - 4: Carrier.Status *(string)*: Carrier, CTL
    - 5: Gene.Dx *(string)*:  mutation status + clinical status
    (“PreSx” suffix = presymptomatic and “Sx” suffix = symptomatic)
    - 6: GLOBALCOG.ZCORE *(float)*: global cognition composite score
    - 7: FTLDCDR_SBL *(int)*: CDR sum of boxes - Clinical Dementia Rating Scale (CDR)
    is a global assessment instrument that yields global and Sum of Boxes (SOB) scores,
    with the global score regularly used in clinical and research settings
    to stage dementia severity. Higher is worse.
    - 8: NFL3_MEAN *(float):* plasma NfL concentrations
    - 9 : ef.unadj.intercept: Executive function unadjusted intercept
    - 10 : ef.unadj.slope: Executive function unadjusted slope
    - 11: ef.adj.intercept: Executive function adjusted intercept
    - 12: ef.adj.slope: Executive function adjusted slope
    - 13: mem.unadj.intercept: Memory unadjusted intercept
    - 14: mem.unadj.slope: Memory unadjusted slope
    - 15: mem.adj.intercept: Memory adjusted intercept
    - 16: mem.adj.slope: Memory adjusted slope
    - 17: disease.age: Disease age

    - 9: HasPlasma? *(int)*: 1, 0 (519 Yes)
    - 19 - 7307: Proteins *(float)*:

    Protein variables are annotated as
      Protein Symbol | UniProt ID^Sequence ID| Matrix (CSF or PLASMA).
      The sequence ID is present only if there is more than one target
      for a given protein: e.g.,
      ABL2|P42684^SL010488@seq.3342.76|PLASMA ,
      ABL2|P42684^SL010488@seq.5261.13|PLASMA

    - 7308: HasCSF? *(int)*: 1, 0 (254 Yes)
    - 7309 - 14597: Proteins *(float)*:
    - 14598 - 15221: Clinical Data - maybe not necessary for right now.

    """

    def __init__(self, root, config, split):
        self.name = "FTD"
        self.root = root
        self.split = split
        self.kfold = config.kfold
        assert self.split in ["train", "val", "test"]

        assert config.sex in SEXES
        assert config.modality in MODALITIES
        assert config.y_val in LABEL_DIM_MAP
        if config.y_val == "carrier":
            assert len(config.mutation) > 1 and "CTL" in config.mutation
            
        # If adj_metric is set to pointcloud, the adjacency matrix is the identity
        # and adj_thresh is irrelevant (by default set to 1)
        config.adj_thresh = 1.0 if config.adj_metric == "pointcloud" else config.adj_thresh

        self.config = config
        self.adj_metric = config.adj_metric
        self.adj_str = f"adj_thresh_{config.adj_thresh}"
        self.y_val_str = f"y_val_{config.y_val}"
        self.num_nodes_str = f"num_nodes_{config.num_nodes}"
        self.mutation_str = f"mutation_{','.join(config.mutation)}"
        self.modality_str = f"{config.modality}"
        self.sex_str = f"sex_{','.join(config.sex)}"
        if self.kfold:
            self.hist_path_str = f"{self.config.y_val}_{self.config.sex}_{self.config.mutation}_{self.config.modality}_random_state_{config.random_state}_{self.config.num_folds}fold_{self.config.fold}_two_pass_{config.two_pass}_histogram.jpg"
            self.orig_hist_path_str = f"{self.config.y_val}_{self.config.sex}_{self.config.mutation}_{self.config.modality}_random_state_{config.random_state}_{self.config.num_folds}fold_{self.config.fold}_two_pass_{config.two_pass}_orig_histogram.jpg"
        else:
            self.hist_path_str = f"{self.config.y_val}_{self.config.sex}_{self.config.mutation}_{self.config.modality}_random_state_{config.random_state}_two_pass_{config.two_pass}_histogram.jpg"
            self.orig_hist_path_str = f"{self.config.y_val}_{self.config.sex}_{self.config.mutation}_{self.config.modality}_random_state_{config.random_state}_two_pass_{config.two_pass}_orig_histogram.jpg"
        super(FTDDataset, self).__init__(
            root, transform=None, pre_transform=None
        )
        self.adj_path = os.path.join(
            self.processed_dir,
            f"adjacency_num_nodes_{config.num_nodes}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_two_pass_{config.two_pass}.csv",
        )
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = LABEL_DIM_MAP[self.config.y_val]
            
        if self.kfold:
            config_tag = f"{self.experiment_id}_random_state_{self.config.random_state}_{self.config.num_folds}fold_{self.config.fold}_two_pass_{config.two_pass}"
            adj_config_tag = f"adjacency_num_nodes_{config.num_nodes}_{self.adj_metric}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_random_state_{self.config.random_state}_{self.config.num_folds}fold_{self.config.fold}_two_pass_{config.two_pass}"
        else:
            config_tag = f"{self.experiment_id}_random_state_{self.config.random_state}_two_pass_{config.two_pass}"
            adj_config_tag = f"adjacency_num_nodes_{config.num_nodes}_{self.adj_metric}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_random_state_{self.config.random_state}_two_pass_{config.two_pass}"
        self.config_tag = config_tag  
        path = os.path.join(
            self.processed_dir,
            f"{config_tag}_{self.split}.pt",
        )
        self.adj_path = os.path.join(
            self.processed_dir,
            f"{adj_config_tag}.csv",
        )
        print("Loading data from:", path)
        self.load(path)

    @property
    def raw_file_names(self):
        """Files that must be present in order to skip downloading them from somewhere.

        Then, the grandparent Dataset class automatically defines raw_paths as:
        raw_path = self.raw_dir + raw_filename
        where: self.processed_dir = self.root + "raw"

        See Also
        --------
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py
        """
        return [self.config.raw_file_name]

    @property
    def processed_file_names(self):
        """Files that must be present in order to skip processing.

        The, the grandparent Dataset class automatically defines processed_paths as:
        processed_path = self.processed_dir + processed_filename
        where: self.processed_dir = self.root + "processed"

        See Also
        --------
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py
        """
        self.experiment_id = f"{self.name}_{self.y_val_str}_{self.adj_metric}_{self.adj_str}_{self.num_nodes_str}_{self.mutation_str}_{self.modality_str}_{self.sex_str}"
        if self.kfold:
            files = [
                f"{self.experiment_id}_random_state_{self.config.random_state}_{self.config.num_folds}fold_{self.config.fold}_two_pass_{self.config.two_pass}_train.pt",
                f"{self.experiment_id}_random_state_{self.config.random_state}_{self.config.num_folds}fold_{self.config.fold}_two_pass_{self.config.two_pass}_val.pt",
            ]
        else:
            files = [
                f"{self.experiment_id}_random_state_{self.config.random_state}_two_pass_{self.config.two_pass}_train.pt",
                f"{self.experiment_id}_random_state_{self.config.random_state}_two_pass_{self.config.two_pass}_val.pt",
                f"{self.experiment_id}_random_state_{self.config.random_state}_two_pass_{self.config.two_pass}_test.pt",
            ]
        print("Processed file names:", files)
        return files

    def create_graph_data(
        self,
        feature,
        label,
        adj_matrix,
        sex,
        mutation,
        age,
    ):
        """Create Data object for each graph.

        Compute attributes x, edge_index, and y for each graph.
        """
        x = feature  # protein concentrations: what is on the nodes
        adj_tensor = torch.tensor(adj_matrix)
        # Find the indices where the matrix has non-zero elements
        pairs_indices = torch.nonzero(adj_tensor, as_tuple=False)
        # Extract the pairs of connected nodes - FIXED: directly transpose without conversion
        edge_index = pairs_indices.t().contiguous()  # Transpose and ensure contiguous memory
        sex = sex.unsqueeze(1)
        mutation = mutation.unsqueeze(1)
        age = age.unsqueeze(1)
        return Data(
            x=x,
            edge_index=edge_index,
            y=label,
            sex=sex,
            mutation=mutation,
            age=age,
        )

    def process(self):
        """
        Read data into huge `Data` list, i.e., a list of graphs.
        """
        (
            features,
            labels,
            protein_columns,
            filtered_sex_col,
            filtered_mutation_col,
            filtered_age_col,
            filtered_did_col,
            filtered_gene_col,
        ) = self.load_csv_data_pre_pt_files(self.config)

        # Convert sex and mutation to categorical labels
        sex_labels = np.array(filtered_sex_col.astype("category").cat.codes)
        mutation_labels = np.array(
            filtered_mutation_col.astype("category").cat.codes
        )
        num_bins = 10
        init_bins = pd.qcut(labels, q=num_bins, labels=False, duplicates="drop")
        # Split data into train and val/test sets
        (
            train_val_features,
            test_features,
            train_val_labels,
            test_labels,
            train_val_sex,
            test_sex,
            train_val_mutation,
            test_mutation,
            train_val_age,
            test_age,
        ) = train_test_split(
            features,
            labels,
            sex_labels,
            mutation_labels,
            filtered_age_col.values,
            test_size=0.2,
            random_state=self.config.random_state,
            stratify=init_bins,
        )

        if self.kfold:
            # Perform k-fold splitting on the train set
            assert self.config.fold < self.config.num_folds, (
                f"Invalid fold index {self.config.fold}, should be lower than the number of folds {self.config.num_folds}"
            )
            num_bins = math.floor(len(train_val_labels) / self.config.num_folds)
            y_binned = pd.qcut(train_val_labels, q=num_bins, labels=False, duplicates="drop")

            skf = StratifiedKFold(
                n_splits=self.config.num_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

            train_index, val_index = next(
                split
                for fold, split in enumerate(skf.split(train_val_features, y_binned))
                if fold == self.config.fold
            )

            train_features = train_val_features[train_index]
            val_features = train_val_features[val_index]
            train_labels = train_val_labels[train_index]
            val_labels = train_val_labels[val_index]
            train_sex = train_val_sex[train_index]
            val_sex = train_val_sex[val_index]
            train_mutation = train_val_mutation[train_index]
            val_mutation = train_val_mutation[val_index]
            train_age = train_val_age[train_index]
            val_age = train_val_age[val_index]

        else:
            test_set = True
            train_features = train_val_features
            train_labels = train_val_labels
            train_sex = train_val_sex
            train_mutation = train_val_mutation
            train_age = train_val_age

            num_bins = 10
            init_bins = pd.qcut(test_labels, q=num_bins, labels=False, duplicates="drop")
            (
                val_features,
                test_features,
                val_labels,
                test_labels,
                val_sex,
                test_sex,
                val_mutation,
                test_mutation,
                val_age,
                test_age,
            ) = train_test_split(
                test_features,
                test_labels,
                test_sex,
                test_mutation,
                test_age,
                test_size=0.5,
                random_state=self.config.random_state,
                stratify=init_bins,
            )
            # Just consider train and test/val splits
            # train_features = train_val_features
            # val_features = test_features
            # train_labels = train_val_labels
            # val_labels = test_labels
            # train_sex = train_val_sex
            # val_sex = test_sex
            # train_mutation = train_val_mutation
            # val_mutation = test_mutation
            # train_age = train_val_age
            # val_age = test_age

        # Unpack the return values from load_csv_data
        if test_set:
            (
                train_features,
                train_labels,
                val_features,
                val_labels,
                train_sex,
                val_sex,
                train_mutation,
                val_mutation,
                train_age,
                val_age,
                adj_matrix,  # This will be a list
                test_features,
                test_labels,
                test_sex,
                test_mutation,
                test_age,
            ) = self.load_csv_data(
                self.config,
                train_features,
                val_features,
                train_labels,
                val_labels,
                train_sex,
                val_sex,
                train_mutation,
                val_mutation,
                train_age,
                val_age,
                test_set,
                test_features,
                test_labels,
                test_sex,
                test_mutation,
                test_age,
            )
        else:
            (
                train_features,
                train_labels,
                val_features,
                val_labels,
                train_sex,
                val_sex,
                train_mutation,
                val_mutation,
                train_age,
                val_age,
                adj_matrix,  # This will be a list
            ) = self.load_csv_data(
                self.config,
                train_features,
                val_features,
                train_labels,
                val_labels,
                train_sex,
                val_sex,
                train_mutation,
                val_mutation,
                train_age,
                val_age,
            )

        train_data_list = []
        val_data_list = []

        # Single adjacency matrix is used for both male and female data
        adj_matrix = adj_matrix

        # Iterate through train data and use the single adjacency matrix
        for feature, label, sex, mutation, age in zip(
            train_features, train_labels, train_sex, train_mutation, train_age
        ):
            data = self.create_graph_data(
                feature, label, adj_matrix, sex, mutation, age
            )
            train_data_list.append(data)

        # Iterate through val data and use the single adjacency matrix
        for feature, label, sex, mutation, age in zip(
            val_features, val_labels, val_sex, val_mutation, val_age
        ):
            data = self.create_graph_data(
                feature, label, adj_matrix, sex, mutation, age
            )
            val_data_list.append(data)

        # Save the train and val data lists
        train_path = f"{self.processed_paths[0]}"
        val_path = f"{self.processed_paths[1]}"
        self.save(train_data_list, train_path)
        self.save(val_data_list, val_path)

        if test_set:
            test_data_list = []
            # Iterate through test data and use the single adjacency matrix
            for feature, label, sex, mutation, age in zip(
                test_features, test_labels, test_sex, test_mutation, test_age   
            ):
                data = self.create_graph_data(
                    feature, label, adj_matrix, sex, mutation, age
                )
                test_data_list.append(data)
            test_path = f"{self.processed_paths[2]}"
            self.save(test_data_list, test_path)

        # # Save the train and val data lists
        # train_path = f"{self.processed_paths[0]}"
        # val_path = f"{self.processed_paths[1]}"
        # self.save(train_data_list, train_path)
        # self.save(val_data_list, val_path)

        # if test_set:
        #     test_data_list = []
        #     # Iterate through test data and use the single adjacency matrix
        #     for feature, label, sex, mutation, age in zip(
        #         test_features, test_labels, test_sex, test_mutation, test_age
        #     ):
        #         data = self.create_graph_data(
        #             feature, label, adj_matrix, sex, mutation, age
        #         )
        #         test_data_list.append(data)
        #     test_path = f"{self.processed_paths[2]}"
        #     self.save(test_data_list, test_path)

    # -----------------------------FUNCTIONS TO GET LABELS---------------------------------#
    def load_y_vals(self, filtered_data):
        """Find the y_val values based on the config."""
        y_vals = filtered_data[Y_VAL_COL_MAP[self.config.y_val]]
        y_vals_mask = ~y_vals.isna()
        y_vals = y_vals[y_vals_mask]

        if self.config.y_val in BINARY_Y_VALS_MAP:
            y_vals = self.load_binary_y_values(y_vals)
        elif self.config.y_val in MULTICLASS_Y_VALS_MAP:
            y_vals = self.load_multiclass_y_values(y_vals)
        # Remove NaN values from y_vals and return filter to remove rows where y_val is NaN

        # Plot histogram of y_vals
        hist_path = os.path.join(
            self.processed_dir,
            self.orig_hist_path_str,
        )
        plot_histogram(
            pd.DataFrame(y_vals), self.config.y_val, save_to=hist_path
        )
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

    # --------------------------- FUNCTIONS PROCESSING ALL --------------------------------#
    def load_csv_data_pre_pt_files(self, config):
        """Load the csv data features and labels. Filter out sex, mutation, modality, and remove erroneous columns."""
        csv_path = self.raw_paths[0]
        print("Loading data from:", csv_path)
        csv_data = pd.read_csv(csv_path)

        # Remove nfl columns
        csv_data = remove_erroneous_columns_and_two_pass_error_proteins(config, csv_data, self.raw_dir)

        # Get the correct subset of proteins based on the mutation, if they have the correct modality measurements, and sex and then use those to find the top proteins and labels
        condition_sex = csv_data[sex_col].isin(self.config.sex)
        condition_modality = csv_data[HAS_MODALITY_COL[self.config.modality]]
        condition_mutation = csv_data[mutation_col].isin(self.config.mutation)
        sex_mutation_modality_filter = (
            condition_sex & condition_mutation & condition_modality
        )
        print(
            "Number of patients with measurements:", condition_modality.sum()
        )
        print(
            f"Number of patients with mutation status in {self.config.mutation}:",
            condition_mutation.sum(),
        )
        print(
            f"Number of patients with sex in {self.config.sex}:",
            condition_sex.sum(),
        )
        print(
            "Total number of patients with all conditions",
            sex_mutation_modality_filter.sum(),
        )
        filtered_data = csv_data[
            sex_mutation_modality_filter
        ]  # Select rows that meet all conditions

        # Extract the y_val values
        y_vals, y_val_mask = self.load_y_vals(filtered_data)
        filtered_data = filtered_data[
            y_val_mask
        ]  # Remove rows where y_val is NaN
        print("final dims of filtered data:", filtered_data.shape)
        # Extract the top proteins (features) for building datasets

        # Filter protein columns based on modality
        protein_cols = [
            col
            for col in filtered_data.columns
            if col.endswith(MODALITY_COL_END[self.config.modality])
        ]
        print("Number of proteins:", len(protein_cols))
        top_proteins = filtered_data[protein_cols]

        # Extract column labels for sex to understand explainer results
        filtered_sex_col = filtered_data[sex_col]
        filtered_mutation_col = filtered_data[mutation_col]
        filtered_age_col = filtered_data[age_col]
        filtered_did_col = filtered_data[did_col]
        filtered_gene_col = filtered_data[gene_col]

        features = np.array(top_proteins, dtype=np.float32)
        labels = np.array(y_vals, dtype=np.float32)

        return (
            features,
            labels,
            protein_cols,
            filtered_sex_col,
            filtered_mutation_col,
            filtered_age_col,
            filtered_did_col,
            filtered_gene_col,
        )  # NOTE: Just returning top_protein_cols to use it in finding top proteins in evaluation.ipynb

    def load_csv_data(
        self,
        config,
        train_features,
        val_features,
        train_labels,
        val_labels,
        train_sex,
        val_sex,
        train_mutation,
        val_mutation,
        train_age,
        val_age,
        test_set=False,
        test_features=None,
        test_labels=None,
        test_sex=None,
        test_mutation=None,
        test_age=None
    ):
        if config.y_val in Y_VALS_TO_NORMALIZE:
            train_labels_norm, train_mean, train_std = log_transform(
                train_labels, train_labels
            )
            save_mean_std(
                train_mean,
                train_std,
                config,
                self.experiment_id,
                self.processed_dir,
            )
            val_labels_norm, val_mean, val_std = log_transform(
                train_labels, val_labels
            )
            if test_set:
                test_labels_norm, test_mean, test_std = log_transform(
                    train_labels, test_labels
                )
                # Plot normalized labels histogram with separate colors for train/val/test
                hist_path = os.path.join(self.processed_dir, self.hist_path_str)
                plot_histogram(
                    [train_labels_norm, val_labels_norm, test_labels_norm],
                    self.config.y_val,
                    save_to=hist_path,
                    data_labels=['train', 'val', 'test']
                )
            else:
                # Plot normalized labels histogram with separate colors for train/val
                hist_path = os.path.join(self.processed_dir, self.hist_path_str)
                plot_histogram(
                    [train_labels_norm, val_labels_norm],
                    self.config.y_val,
                    save_to=hist_path,
                    data_labels=['train', 'val']
                )
        else:
            # If not normalizing, just use the original labels (ensure float32)
            train_labels_norm = train_labels.astype(np.float32) if isinstance(train_labels, np.ndarray) else train_labels
            val_labels_norm = val_labels.astype(np.float32) if isinstance(val_labels, np.ndarray) else val_labels
            if test_set:
                test_labels_norm = test_labels.astype(np.float32) if isinstance(test_labels, np.ndarray) else test_labels

        train_features_for_adj = train_features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features).astype(np.float32)
        val_features = scaler.transform(val_features).astype(np.float32)
        if test_set:
            test_features = scaler.transform(test_features).astype(np.float32)
        train_age = scaler.fit_transform(train_age.reshape(-1, 1)).astype(np.float32)
        val_age = scaler.transform(val_age.reshape(-1, 1)).astype(np.float32)
        if test_set:
            test_age = scaler.transform(test_age.reshape(-1, 1)).astype(np.float32)
        train_sex = scaler.fit_transform(train_sex.reshape(-1, 1)).astype(np.float32)
        val_sex = scaler.transform(val_sex.reshape(-1, 1)).astype(np.float32)
        if test_set:
            test_sex = scaler.transform(test_sex.reshape(-1, 1)).astype(np.float32)
        train_mutation = scaler.fit_transform(train_mutation.reshape(-1, 1)).astype(np.float32)
        val_mutation = scaler.transform(val_mutation.reshape(-1, 1)).astype(np.float32)
        if test_set:
            test_mutation = scaler.transform(test_mutation.reshape(-1, 1)).astype(np.float32)

        train_features = torch.FloatTensor(
            train_features.reshape(-1, train_features.shape[1], 1)
        )
        val_features = torch.FloatTensor(
            val_features.reshape(-1, val_features.shape[1], 1)
        )
        if test_set:
            test_features = torch.FloatTensor(
                test_features.reshape(-1, test_features.shape[1], 1)
            )
        train_labels_norm = torch.FloatTensor(train_labels_norm)
        val_labels_norm = torch.FloatTensor(val_labels_norm)
        if test_set:
            test_labels_norm = torch.FloatTensor(test_labels_norm)
        train_sex = torch.FloatTensor(train_sex)
        val_sex = torch.FloatTensor(val_sex)
        if test_set:
            test_sex = torch.FloatTensor(test_sex)
        train_mutation = torch.FloatTensor(train_mutation)
        val_mutation = torch.FloatTensor(val_mutation)
        if test_set:
            test_mutation = torch.FloatTensor(test_mutation)
        train_age = torch.FloatTensor(train_age)
        val_age = torch.FloatTensor(val_age)
        if test_set:
            test_age = torch.FloatTensor(test_age)

        print(
            "Training features and labels:",
            train_features.shape,
            train_labels_norm.shape,
        )
        print(
            "Training sex, mutation and age labels shape:",
            train_sex.shape,
            train_mutation.shape,
            train_age.shape,
        )
        print("Val features and labels:", val_features.shape, val_labels_norm.shape)
        print(
            "Val sex, mutation and age labels shape:",
            val_sex.shape,
            val_mutation.shape,
            val_age.shape,
        )
        if test_set:
            print(
                "Test features and labels:",
                test_features.shape,
                test_labels_norm.shape,
            )
            print(
                "Test sex, mutation and age labels shape:",
                test_sex.shape,
                test_mutation.shape,
                test_age.shape,
            )
        # Calculate adjacency matrix
        if self.kfold:
            adj_path = os.path.join(
                self.processed_dir,
                f"adjacency_num_nodes_{config.num_nodes}_{self.adj_metric}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_random_state_{self.config.random_state}_{self.config.num_folds}fold_{self.config.fold}_two_pass_{config.two_pass}.csv",
            )
        else:
            adj_path = os.path.join(
                self.processed_dir,
                f"adjacency_num_nodes_{config.num_nodes}_{self.adj_metric}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_random_state_{self.config.random_state}_two_pass_{config.two_pass}.csv",
            )
            
        self.adj_path = adj_path
        adj_matrix = None
        # Calculate and save adjacency matrix
        if not os.path.exists(adj_path):
            adj_matrix = compute_adjacency_matrix(
                config, train_features_for_adj, save_to=adj_path
            )
        # get the adjacency matrix after applying the threshold
        adj_matrix = self.get_adjacency_matrix(
            adj_path, config.adj_thresh, config, adj_matrix=adj_matrix
        )
        # Plot and save adjacency matrix as jpg
        # self.plot_adj_matrix(
        #     adj_matrix,
        #     os.path.join(
        #         self.processed_dir,
        #         f"adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_{config.num_folds}fold_{config.fold}.jpg",
        #     ),
        # )
        if test_set:
            return (
                train_features,
                train_labels_norm,
                val_features,
                val_labels_norm,
                train_sex,
                val_sex,
                train_mutation,
                val_mutation,
                train_age,
                val_age,
                adj_matrix,
                test_features,
                test_labels_norm,
                test_sex,
                test_mutation,
                test_age,
            )
        else:
            return (
                train_features,
                train_labels_norm,
                val_features,
                val_labels_norm,
                train_sex,
                val_sex,
                train_mutation,
                val_mutation,
                train_age,
                val_age,
                adj_matrix,
            )

    def get_adjacency_matrix(self, path, adj_thresh, config, adj_matrix=None):
        """
        Load and threshold an adjacency matrix.

        Parameters:
        - path: Path to the CSV file containing the adjacency matrix.
        - adj_thresh: Threshold value to convert the matrix to a binary form.

        Returns:
        - adj_matrix: Thresholded adjacency matrix as a torch FloatTensor.
        """
        if adj_matrix is None:
            print(f"Loading adjacency matrix from: {path}...")
            adj_matrix = np.array(pd.read_csv(path, header=None)).astype(float)
        adj_matrix = torch.FloatTensor(
            np.where(adj_matrix >= adj_thresh, 1, 0)
        )  # Thresholding
        print("Adjacency matrix shape:", adj_matrix.shape)
        expected_shape = (
            config.num_nodes,
            config.num_nodes,
        )
        # Assert the shape matches the expected shape
        assert adj_matrix.shape == expected_shape, (
            f"Unexpected shape: {adj_matrix.shape}. Expected shape: {expected_shape}"
        )
        print("Number of edges:", adj_matrix.sum())
        return adj_matrix

    def plot_adj_matrix(self, adj_matrix, path):
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", ["white", "black"]
        )
        plt.figure()
        plt.imshow(adj_matrix.cpu().numpy(), cmap=cmap)
        plt.colorbar(ticks=[0, 1], label="Adjacency Value")
        plt.title("Visualization of Adjacency Matrix")
        plt.savefig(path)
        plt.close()


def remove_erroneous_columns_and_two_pass_error_proteins(config, csv_data, raw_dir):
    """Remove columns that have bimodal distributions, keeping only two-pass proteins if specified."""
    csv_path = os.path.join(raw_dir, config.error_protein_file_name)
    error_proteins_df = pd.read_excel(csv_path)
    # Extract column names under "CSF"
    csf_columns = error_proteins_df['CSF'].dropna().tolist()
    columns_to_remove = list(set(csf_columns))
    
    # Handle two-pass error proteins if config.two_pass is True
    if config.two_pass:
        two_pass_csv_path = os.path.join(raw_dir, config.two_pass_error_protein_file_name)
        two_pass_error_proteins_df = pd.read_csv(two_pass_csv_path)
        # Get the first column (rows) and append "|CSF" to each
        two_pass_columns_to_keep = [row + "|CSF" for row in two_pass_error_proteins_df.iloc[:, 0].dropna()]
        
        # Remove any two-pass proteins from the original error protein removal list
        columns_to_remove = [col for col in columns_to_remove if col not in two_pass_columns_to_keep]
        
        # Get all protein columns (those ending with |CSF or |PLASMA)
        all_protein_columns = [col for col in csv_data.columns if col.endswith('|CSF') or col.endswith('|PLASMA')]
        
        # Remove all protein columns EXCEPT the two-pass ones
        columns_to_remove.extend([col for col in all_protein_columns if col not in two_pass_columns_to_keep])
    
    # Always remove NFL columns if y_val is "nfl" (regardless of two_pass setting)
    if config.y_val == "nfl":
        columns_to_remove.extend(
            ['NEFL|P07196|CSF', 'NEFH|P12036|CSF', 'NEFL|P07196|PLASMA', 'NEFH|P12036|PLASMA']
        )
    
    # Remove the columns
    csv_data = csv_data.drop(columns=columns_to_remove, errors='ignore')
    return csv_data



def compute_adjacency_matrix(config, dataset, save_to):
    """
    Compute the adjacency matrix for a given dataset using R's igraph package.

    Parameters
    ----------
    config : dict
        Configuration parameters for the dataset.
    dataset : object
        The dataset object containing the graph data.
    save_to : str
        Path to save the computed adjacency matrix.

    Returns
    -------
    None
    """
    if config["adj_metric"] == 'wgcna':
        adjacency_matrix = calculate_wgcna_matrix(config, dataset)
    elif config["adj_metric"] == 'mutual_information':
        adjacency_matrix = calculate_mutual_information_matrix(dataset)
    elif config["adj_metric"] == 'spearman_correlation':
        adjacency_matrix = calculate_spearman_correlation_matrix(dataset)
    elif config["adj_metric"] == 'distance_correlation':
        adjacency_matrix = calculate_distance_correlation_matrix(dataset)
    elif config["adj_metric"] == 'pointcloud':
        adjacency_matrix = np.zeros((config["num_nodes"], config["num_nodes"]))
    # elif config["adj_metric"] == 'maximal_information_coefficient':
    #     calculate_maximal_information_coefficient_matrix(dataset)
    else:
        raise ValueError(f"Unknown adjacency metric: {config['adj_metric']}")
    
    # Set diagonal to zero to avoid self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    #Exept in pointcloud setting, where we want to keep the diagonal as 1
    if config["adj_metric"] == 'pointcloud':
        np.fill_diagonal(adjacency_matrix, 1.0)
    # Normalize by the maximum value (avoid division by zero)
    max_val = adjacency_matrix.max()
    if max_val > 0:
        adjacency_matrix = adjacency_matrix / max_val
    
    # Save the adjacency matrix to the specified file path
    adjacency_df = pd.DataFrame(adjacency_matrix)
    with open(save_to, "w") as f:
        adjacency_df.to_csv(f, header=None, index=False)
    print(f"Adjacency matrix saved to: {save_to}")
    
    return adjacency_matrix
    
    

# ----------------------- HELPER FUNCTIONS--------------------------


def plot_histogram(data, x_label, save_to, data_labels=None):
    """
    Plot histogram with support for multiple datasets with different colors.
    
    Parameters:
    - data: Single dataset (DataFrame/array) or list of datasets
    - x_label: Label for x-axis
    - save_to: Path to save the plot
    - data_labels: Optional list of labels for each dataset (e.g., ['train', 'val', 'test'])
    """
    if isinstance(data, list):
        # Multiple datasets - plot with different colors
        colors = ['blue', 'orange', 'green']
        for i, (dataset, label) in enumerate(zip(data, data_labels or [''] * len(data))):
            plt.hist(dataset, bins=30, alpha=0.5, label=label, color=colors[i % len(colors)])
        if data_labels:
            plt.legend()
    else:
        # Single dataset - original behavior
        plt.hist(data, bins=30, alpha=0.5)
    
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {x_label}")
    plt.savefig(save_to, format="jpg")
    plt.close()


def log_transform(train_data, data, log=False):
    if log:
        # Log transformation
        data = np.log(data)
    mean = np.mean(train_data)
    std = np.std(train_data)
    standardized_log_data = (data - mean) / std
    # Ensure float32 dtype to avoid Double/Float mismatch in PyTorch
    standardized_log_data = standardized_log_data.astype(np.float32)
    return standardized_log_data, mean, std


def reverse_log_transform(standardized_log_data, mean, std, log=False):
    # De-standardize the data

    data = standardized_log_data * std + mean
    if log:
        data = torch.exp(data)
    return data


def save_mean_std(mean, std, config, experiment_id, processed_dir):
    if config.kfold:
        file_name = f"{experiment_id}_train_random_state_{config.random_state}_{config.num_folds}fold_{config.fold}.json"
    else:
        file_name = (
            f"{experiment_id}_train_random_state_{config.random_state}.json"
        )
    file_path = os.path.join(processed_dir, file_name)

    with open(file_path, "w") as f:
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")
    print(f"Mean and std saved to: {file_path}")
