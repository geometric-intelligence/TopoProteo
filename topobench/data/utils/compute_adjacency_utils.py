import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import r as R
from rpy2.robjects.packages import importr
import os
os.environ["RPY2_RINTERFACE_SIGINT"] = "FALSE"
import PyWGCNA
import pandas as pd
import numpy as np
import torch

from joblib import Parallel, delayed
from npeet import entropy_estimators as ee
from torchmetrics.functional import signal_noise_ratio
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import dcor
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
# from minepy import MINE

    
def calculate_wgcna_matrix(config, dataset):
    """
    Calculate the correlation matrix for the dataset using rpy2 and save it to a file.

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
    np.ndarray
        The computed adjacency matrix based on WGCNA
    """
    pandas2ri.activate()
    wgcna = importr("WGCNA")
    # Convert the input `protein_data` (NumPy array) to a pandas DataFrame if necessary
    if not isinstance(dataset, pd.DataFrame):
        protein_data_df = pd.DataFrame(dataset)
    else:
        protein_data_df = dataset

    # Convert pandas DataFrame to R DataFrame
    r_protein_data = pandas2ri.py2rpy(protein_data_df)

    # Call R's `pickSoftThreshold` function to determine the soft thresholding power
    soft_threshold_result = wgcna.pickSoftThreshold(
        r_protein_data, corFnc="bicor", networkType="signed"
    )
    soft_threshold_power = soft_threshold_result.rx2("powerEstimate")[
        0
    ]  # Extract the estimated power
    if config.modality == "csf" and all(
        mut in config.mutation for mut in ["C9orf72", "MAPT", "GRN", "CTL"]
    ):
        soft_threshold_power = 9  # went over values w Rowan and selected this.
    # print(f"Soft threshold power: {soft_threshold_power}")

    # Call R's `adjacency` function using the chosen power and the desired correlation function (bicor)
    adjacency_matrix_r = wgcna.adjacency(
        r_protein_data,
        power=soft_threshold_power,
        type="signed",  # Specify network type
        corFnc="bicor",  # Use biweight midcorrelation
    )

    # Convert the resulting adjacency matrix from R back to a NumPy array
    adjacency_matrix = adjacency_matrix_r

    return adjacency_matrix

    
def calculate_mutual_information_matrix(dataset):
    """
    Calculate the mutual information matrix for the dataset and save it to a file.

    Parameters
    ----------
    dataset : object
        The dataset object containing the graph data.
        
    Returns
    -------
    np.ndarray
        The computed adjacency matrix based on mutual information
    """
    protein_data = dataset.T
    # adjacency_matrix = pairwise_distances(dataset.T, metric=mutual_info_score, n_jobs=-1)
    # dist_matrix = squareform(pdist(protein_data, metric=mutual_info_score))
    num_proteins = protein_data.shape[0]
    adjacency_matrix = np.zeros((num_proteins, num_proteins))
    # for i in range(num_proteins):
    #     for j in range(i+1,num_proteins):
    #         # Calculate mutual information between protein i and protein j across the samples
    #         score = ee.mi(protein_data[i], protein_data[j], k=3)  # Using k=3 for nearest neighbors
    #         adjacency_matrix[i, j] = score
    #         adjacency_matrix[j, i] = score  # Ensure symmetry
    a = 1
    # Define a function to compute mutual information for a pair (i, j)
    def compute_mi(i, j):
        score = ee.mi(protein_data[i], protein_data[j], k=3)
        return (i, j, score)

    # Prepare all unique pairs (i, j) with i < j
    pairs = [(i, j) for i in range(num_proteins) for j in range(i + 1, num_proteins)]

    # Compute in parallel
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_mi)(i, j) for i, j in pairs
    )

    # Fill the adjacency matrix
    for i, j, score in results:
        adjacency_matrix[i, j] = score
        adjacency_matrix[j, i] = score  # Ensure symmetry
    
    return adjacency_matrix
    

def calculate_distance_correlation_matrix(dataset):
    """
    Calculate the distance correlation matrix for the dataset and save it to a file.

    Parameters
    ----------
    dataset : object
        The dataset object containing the graph data.
        
    Returns
    -------
    np.ndarray
        The computed adjacency matrix based on distance correlation
    """
    protein_data = dataset.T
    num_proteins = protein_data.shape[0]
    adjacency_matrix = np.zeros((num_proteins, num_proteins))
    # Define a function to compute mutual information for a pair (i, j)
    def compute_dcor(i, j):
        score = dcor.distance_correlation(protein_data[i], protein_data[j])
        return (i, j, score)

    # Prepare all unique pairs (i, j) with i < j
    pairs = [(i, j) for i in range(num_proteins) for j in range(i + 1, num_proteins)]

    # Compute in parallel
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_dcor)(i, j) for i, j in pairs
    )

    # Fill the adjacency matrix
    for i, j, score in results:
        adjacency_matrix[i, j] = score
        adjacency_matrix[j, i] = score  # Ensure symmetry
    
    return adjacency_matrix

    
def calculate_spearman_correlation_matrix(dataset):
    """
    Calculate the Spearman correlation matrix for the dataset and save it to a file.
    
    Parameters
    ----------
    dataset : object
        The dataset object containing the graph data.
        
    Returns
    -------
    np.ndarray
        The computed adjacency matrix based on Spearman correlation
    """
    adjacency_matrix, _ = spearmanr(dataset, axis=0)
    return adjacency_matrix
    
    
# def calculate_maximal_information_coefficient_matrix(config, dataset, save_to):
#     """
#     Calculate the maximal information coefficient matrix for the dataset and save it to a file.

#     Parameters
#     ----------
#     config : dict
#         Configuration parameters for the dataset.
#     dataset : object
#         The dataset object containing the graph data.
#     save_to : str
#         Path to save the computed adjacency matrix.
#     """
    
#     protein_data = dataset.T
#     num_proteins = protein_data.shape[0]
#     adjacency_matrix = np.zeros((num_proteins, num_proteins))
#     # Initialize MINE estimator
#     mine = MINE(alpha=0.6, c=15)
    
#     # Define a function to compute MIC for a pair (i, j)
#     def compute_mic(i, j):
#         mine.compute_score(protein_data[i], protein_data[j])
#         score = mine.mic()
#         return (i, j, score)

#     # Prepare all unique pairs (i, j) with i < j
#     pairs = [(i, j) for i in range(num_proteins) for j in range(i + 1, num_proteins)]

#     # Compute in parallel
#     results = Parallel(n_jobs=-1, prefer="threads")(
#         delayed(compute_mic)(i, j) for i, j in pairs
#     )

#     # Fill the adjacency matrix
#     for i, j, score in results:
#         adjacency_matrix[i, j] = score
#         adjacency_matrix[j, i] = score  # Ensure symmetry
    
#     # Save the adjacency matrix to the specified file path
#     save_adjacency_matrix(adjacency_matrix, save_to)


# def calculate_signal_noise_ratio_matrix(config, dataset, save_to):
#     """
#     Calculate the signal-to-noise ratio matrix for the dataset and save it to a file.

#     Parameters
#     ----------
#     config : dict
#         Configuration parameters for the dataset.
#     dataset : object
#         The dataset object containing the graph data.
#     save_to : str
#         Path to save the computed adjacency matrix.
#     """
#     protein_data = dataset.T
#     num_proteins = protein_data.shape[0]
#     adjacency_matrix = np.zeros((num_proteins, num_proteins))
#     for i in range(num_proteins):
#         for j in range(num_proteins):
#             if i != j:  # Avoid self-comparison
#                 # Convert to torch tensors before passing to signal_noise_ratio
#                 x = torch.tensor(protein_data[i], dtype=torch.float32)
#                 y = torch.tensor(protein_data[j], dtype=torch.float32)
#                 adjacency_matrix[i, j] = signal_noise_ratio(x, y).item()
#     # Save the adjacency matrix to the specified file path
#     save_adjacency_matrix(adjacency_matrix, save_to)


 
# def calculate_adjacency_matrix(config, protein_data, save_to):
#     import rpy2.robjects as ro
#     from rpy2.robjects import pandas2ri
#     from rpy2.robjects import r as R
#     from rpy2.robjects.packages import importr

#     pandas2ri.activate()
#     wgcna = importr("WGCNA")
#     """Calculate and save adjacency matrix using R's WGCNA."""
#     # Convert the input `protein_data` (NumPy array) to a pandas DataFrame if necessary
#     if not isinstance(protein_data, pd.DataFrame):
#         protein_data_df = pd.DataFrame(protein_data)
#     else:
#         protein_data_df = protein_data

#     # Convert pandas DataFrame to R DataFrame
#     r_protein_data = pandas2ri.py2rpy(protein_data_df)

#     # Call R's `pickSoftThreshold` function to determine the soft thresholding power
#     soft_threshold_result = wgcna.pickSoftThreshold(
#         r_protein_data, corFnc="bicor", networkType="signed"
#     )
#     soft_threshold_power = soft_threshold_result.rx2("powerEstimate")[
#         0
#     ]  # Extract the estimated power
#     if config.modality == "csf" and all(
#         mut in config.mutation for mut in ["C9orf72", "MAPT", "GRN", "CTL"]
#     ):
#         soft_threshold_power = 9  # went over values w Rowan and selected this.
#     print(f"Soft threshold power: {soft_threshold_power}")

#     # Call R's `adjacency` function using the chosen power and the desired correlation function (bicor)
#     adjacency_matrix_r = wgcna.adjacency(
#         r_protein_data,
#         power=soft_threshold_power,
#         type="signed",  # Specify network type
#         corFnc="bicor",  # Use biweight midcorrelation
#     )

#     # Convert the resulting adjacency matrix from R back to a NumPy array
#     adjacency_matrix = adjacency_matrix_r

#     print("Adjacency matrix shape:", adjacency_matrix.shape)

#     # Save the adjacency matrix to the specified file path
#     adjacency_df = pd.DataFrame(adjacency_matrix)
#     with open(save_to, "w") as f:
#         adjacency_df.to_csv(f, header=None, index=False)
#     print(f"Adjacency matrix saved to: {save_to}")