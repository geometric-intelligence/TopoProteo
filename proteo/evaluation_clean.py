"""
Clean, refactored evaluation module for protein importance analysis.

This module provides tools for:
- Loading trained models and running explainer analysis
- Creating protein importance visualizations
- Performing PCA analysis on importance scores
- Comparing results across different models
"""

import os
import types
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_geometric
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict, Counter
import re

# Imports for dimensionality reduction and clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom imports from proteo
from topobench.data.datasets.ftd_dataset import FTDDataset
from topobench.data.utils.utils import construct_datasets
import plotly.graph_objs as go
from matplotlib.lines import Line2D
from tutorials.utils_final_results import load_model_checkpoint
from topobench.model.model import TBModel
from omegaconf import OmegaConf
from hydra.utils import instantiate


@dataclass
class ExplainerConfig:
    """Configuration for explainer analysis."""
    algorithm: str = 'IntegratedGradients'
    explanation_type: str = 'model'
    node_mask_type: str = 'attributes'
    edge_mask_type: Optional[str] = None
    threshold_type: str = 'topk'
    top_k: int = 100
    mode: str = 'regression'
    task_level: str = 'graph'
    return_type: str = 'raw'

@dataclass
class PlotConfig:
    """Configuration for plotting parameters."""
    figure_size: Tuple[int, int] = (28, 10)
    dpi: int = 300
    bar_width: float = 0.6
    top_n: int = 100
    font_size: Dict[str, int] = None
    
    def __post_init__(self):
        if self.font_size is None:
            self.font_size = {
                'title': 24,
                'label': 18,
                'tick': 14,
                'annotation': 6
            }

class ModelLoader:
    """Handles model loading and configuration management."""
    
    @staticmethod
    def load_checkpoint(ckpt_path: str, map_location: str = "cpu") -> Tuple[Any, Any]:
        """
        Load model checkpoint and return model with config.
        
        Parameters
        ----------
        ckpt_path : str
            Path to the checkpoint file
        map_location : str
            Device to load the checkpoint on
            
        Returns
        -------
        Tuple[Any, Any]
            Model and configuration objects
        """
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=map_location)
        hp = ckpt.get("hyper_parameters", {})
        
        if "cfg" not in hp or not isinstance(hp["cfg"], str):
            raise KeyError("Checkpoint missing YAML config under 'hyper_parameters[\"cfg\"]'.")
        
        # Fix YAML parsing issue
        cfg_yaml = hp["cfg"].replace(": None", ": null")
        cfg = OmegaConf.create(cfg_yaml)
        
        # Build model
        model = instantiate(
            cfg.model,
            evaluator=cfg.evaluator,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
            cfg_yaml=cfg_yaml,
        )
        
        # Load weights
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print("missing:", missing)
            print("unexpected:", unexpected)
        
        # Modify forward method for explainer compatibility
        ModelLoader._modify_forward_method(model, cfg)
        
        model.eval()
        return model, cfg
    
    @staticmethod
    def _modify_forward_method(model, cfg):
        """Modify forward method for explainer compatibility."""
        orig_forward = model.forward
        
        def pred_only_forward(self, x, edge_index, data, **kwargs):
            """Forward method that creates proper batch for explainer."""
            batch_data = torch_geometric.data.Data(
                x_0=x, 
                edge_index=edge_index, 
                batch_0=torch.zeros(x.size(0), dtype=torch.int64, device=x.device),
                sex=data.sex, 
                mutation=data.mutation, 
                age=data.age,
                y=data.y if hasattr(data, 'y') else None
            )
            batch = torch_geometric.data.Batch.from_data_list([batch_data])
            out = orig_forward(batch)
            
            if isinstance(out, dict):
                return out.get("logits", next(v for v in out.values() if torch.is_tensor(v)))
            if isinstance(out, (tuple, list)):
                return next(v for v in out if torch.is_tensor(v))
            return out
        
        model.forward = types.MethodType(pred_only_forward, model)
    
    @staticmethod
    def to_legacy_config(cfg) -> Any:
        """Convert nested Hydra config to flat config."""
        p = cfg.dataset.loader.parameters
        
        flat = {
            "data_dir": p.get("data_dir"),
            "random_state": p.get("random_state", cfg.get("seed", 0)),
            "kfold": p.get("kfold", False),
            "num_folds": p.get("num_folds", 5),
            "fold": p.get("fold", 0),
            "raw_file_name": p.raw_file_name,
            "error_protein_file_name": p.error_protein_file_name,
            "y_val": p.y_val,
            "modality": p.modality,
            "mutation": p.mutation,
            "sex": p.sex,
            "num_nodes": p.num_nodes,
            "adj_metric": p.adj_metric,
            "adj_thresh": p.adj_thresh,
            "wgcna_minModuleSize": p.get("wgcna_minModuleSize", 10),
            "wgcna_mergeCutHeight": p.get("wgcna_mergeCutHeight", 0.25),
        }
        
        if flat["adj_metric"] == "pointcloud":
            flat["adj_thresh"] = 1.0
        
        return OmegaConf.create(flat)

class DataProcessor:
    """Handles data loading and preprocessing for explainer analysis."""
    
    def __init__(self, config):
        self.config = config
        self.root = config.data_dir
        self.random_state = config.random_state
    
    def load_datasets(self) -> Tuple[Any, Any]:
        """Load train and test datasets."""
        return construct_datasets(self.config)
    
    def get_demographics_and_protein_ids(self) -> Tuple[np.ndarray, ...]:
        """Get demographic information (sex, mutation, age, etc.)."""
        train_dataset = FTDDataset(self.root, self.config, "train")
        _, _, top_protein_columns, filtered_sex_col, filtered_mutation_col, filtered_age_col, filtered_did_col, filtered_gene_col = train_dataset.load_csv_data_pre_pt_files(self.config)
        
        # Split data
        train_sex_labels, test_sex_labels, train_mutation_labels, test_mutation_labels, train_age_labels, test_age_labels, train_did_labels, test_did_labels, train_gene_col, test_gene_col = train_test_split(
            filtered_sex_col, filtered_mutation_col, filtered_age_col, filtered_did_col, filtered_gene_col, 
            test_size=0.20, random_state=self.random_state
        )
        
        # Combine train and test
        total_sex_labels = np.concatenate((train_sex_labels, test_sex_labels))
        total_mutation_labels = np.concatenate((train_mutation_labels, test_mutation_labels))
        total_age_labels = np.concatenate((train_age_labels, test_age_labels))
        total_did_labels = np.concatenate((train_did_labels, test_did_labels))
        total_gene_labels = np.concatenate((train_gene_col, test_gene_col))
        
        return np.array(top_protein_columns), (total_sex_labels, total_mutation_labels, total_age_labels, 
                total_did_labels, train_did_labels, test_did_labels, total_gene_labels)
    
    def get_baseline_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get baseline data for explainer."""
        train_dataset = FTDDataset(self.root, self.config, "train")
        features, _, top_protein_columns, _, _, _, _, _ = train_dataset.load_csv_data_pre_pt_files(self.config)
        train_features, test_features = train_test_split(features, test_size=0.20, random_state=self.random_state)
        combined_features = np.concatenate((train_features, test_features), axis=0)
        
        # Get CTL baseline
        scaler = StandardScaler()
        scaler.fit(train_features)
        
        csv_data = pd.read_csv(train_dataset.raw_paths[0])
        condition_ctl = csv_data["Mutation"].isin(["CTL"])
        ctl_data = csv_data[condition_ctl]
        proteins_ctl = ctl_data[top_protein_columns].dropna()
        proteins_ctl_scaled = scaler.transform(proteins_ctl)
        baseline_mean = proteins_ctl_scaled.mean(axis=0)
        
        return baseline_mean, combined_features

class Plotter:
    """Handles all plotting functionality."""
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
    
    def plot_importance_scores(self, explanations: List, labels: List, 
                              filename: str, title: str, ylabel: str) -> None:
        """Plot importance scores for multiple samples."""
        plt.figure()
        for i, importance in enumerate(explanations):
            plt.plot(sorted(importance, reverse=True), label=f'Person {i}')
        
        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                  fontsize='small', ncol=1)
        plt.xlabel('Protein')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        plot_filename = os.path.join('explainer_plots', filename)
        plt.show()
    
    def plot_bar_chart(self, protein_dict: Dict, title: str, x_label: str, 
                      y_label: str, filename: Optional[str] = None, 
                      top_n: int = None) -> None:
        """Create bar charts for protein importance."""
        top_n = top_n or self.config.top_n
        
        # Sort and get top/bottom
        sorted_items_desc = dict(sorted(protein_dict.items(), key=lambda item: item[1], reverse=True))
        sorted_items_asc = dict(sorted(protein_dict.items(), key=lambda item: item[1]))
        
        top_highest = dict(list(sorted_items_desc.items())[:top_n])
        top_lowest = dict(list(sorted_items_asc.items())[:top_n])
        
        # Plot highest
        self._plot_single_bar_chart(top_highest, f"Top {top_n} Highest - {title}", 
                                   x_label, y_label, 'lightcoral', 
                                   f"{filename}_highest.png" if filename else None)
        
        # Plot lowest
        self._plot_single_bar_chart(top_lowest, f"Top {top_n} Lowest - {title}", 
                                   x_label, y_label, 'skyblue', 
                                   f"{filename}_lowest.png" if filename else None)
    
    def _plot_single_bar_chart(self, data: Dict, title: str, x_label: str, 
                              y_label: str, color: str, filename: Optional[str] = None) -> None:
        """Create a single bar chart."""
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        plt.bar(data.keys(), data.values(), color=color, width=self.config.bar_width)
        plt.xlabel(x_label, fontsize=self.config.font_size['label'])
        plt.ylabel(y_label, fontsize=self.config.font_size['label'])
        plt.title(title, fontsize=self.config.font_size['title'])
        plt.xticks(rotation=90, ha='right', fontsize=12)
        plt.yticks(fontsize=self.config.font_size['tick'])
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        plt.show()
    
    def plot_top_bar_chart(self, protein_dict: Dict, title: str, x_label: str, 
                          y_label: str, filename: Optional[str] = None, top_n: int = None) -> None:
        """Create a bar chart for top N highest values."""
        top_n = top_n or self.config.top_n
        
        # Sort and get top N highest
        sorted_items_desc = dict(sorted(protein_dict.items(), key=lambda item: item[1], reverse=True))
        top_highest = dict(list(sorted_items_desc.items())[:top_n])
        
        # Create the plot
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        plt.bar(top_highest.keys(), top_highest.values(), color='skyblue', width=self.config.bar_width)
        plt.xlabel(x_label, fontsize=self.config.font_size['label'])
        plt.ylabel(y_label, fontsize=self.config.font_size['label'])
        plt.title(f"Top {top_n} - {title}", fontsize=self.config.font_size['title'])
        plt.xticks(rotation=90, ha='right', fontsize=12)
        plt.yticks(fontsize=self.config.font_size['tick'])
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{filename}_top_{top_n}.png")
        plt.show()

class ExplainerAnalyzer:
    """Handles explainer analysis and results processing."""
    
    def __init__(self, model, config: ExplainerConfig, device: str = 'cuda', plotter: Plotter = None):
        self.model = model
        self.config = config
        self.device = device
        self.plotter = plotter or Plotter()
        self.explainer = self._create_explainer()
    
    def _create_explainer(self) -> Explainer:
        """Create explainer instance."""
        return Explainer(
            model=self.model.to(self.device),
            algorithm=CaptumExplainer(self.config.algorithm),
            explanation_type=self.config.explanation_type,
            model_config=dict(
                mode=self.config.mode,
                task_level=self.config.task_level,
                return_type=self.config.return_type
            ),
            node_mask_type=self.config.node_mask_type,
            edge_mask_type=self.config.edge_mask_type,
            threshold_config=dict(
                threshold_type=self.config.threshold_type,
                value=self.config.top_k,
            ),
        )
    
    def analyze_dataset(self, dataset: List, protein_ids: np.ndarray, 
                       patient_ids: Union[pd.Series, List], filename: str) -> Dict[str, Any]:
        """Analyze a single dataset and return comprehensive results."""
        n_people = len(dataset)
        n_nodes = len(dataset[0].x)
        print(f"n_people = {n_people}, n_nodes = {n_nodes}")
        
        # Initialize tracking dictionaries
        results = self._initialize_tracking_dicts(protein_ids)
        
        # Process each sample
        for i, data in enumerate(dataset):
            explanation = self.explainer(data.x, data.edge_index, data=data, target=None, index=None)
            node_importance = np.array(explanation.node_mask.cpu().detach().numpy()).flatten()
            
            if all(importance == 0 for importance in node_importance):
                print(f"Warning: Person {i} has a node importance list with all zeros.")
            
            # Process this sample's results
            self._process_sample_results(i, node_importance, protein_ids, patient_ids, results)
        
        # Create plots
        self._create_importance_plots(results, filename)
        
        # Add protein count
        all_important_proteins = [protein for top_proteins in results['all_top_proteins'] for protein in top_proteins]
        results['protein_count'] = Counter(all_important_proteins)
        
        return results
    
    def _initialize_tracking_dicts(self, protein_ids: np.ndarray) -> Dict[str, Any]:
        """Initialize dictionaries for tracking results."""
        return {
            'sum_node_importance_raw': {protein_id: 0 for protein_id in protein_ids},
            'sum_node_importance_percent': {protein_id: 0 for protein_id in protein_ids},
            'positive_percent_by_protein': {protein_id: 0 for protein_id in protein_ids},
            'negative_percent_by_protein': {protein_id: 0 for protein_id in protein_ids},
            'all_raw_importances': [],
            'all_percent_importances': [],
            'all_top_proteins': [],
            'all_labels': []
        }
    
    def _process_sample_results(self, i: int, node_importance: np.ndarray, 
                              protein_ids: np.ndarray, patient_ids: pd.Series, 
                              results: Dict[str, Any]):
        """Process results for a single sample."""
        # Store raw importance
        results['all_raw_importances'].append(node_importance.tolist())
        
        # Convert to percentages
        total_importance = np.sum(np.abs(node_importance))
        importance_percentages = (node_importance / total_importance) * 100
        results['all_percent_importances'].append(importance_percentages)
        
        # Calculate positive/negative contributions
        total_positive = np.sum(node_importance[node_importance > 0])
        total_negative = np.sum(np.abs(node_importance[node_importance < 0]))
        
        # Get top proteins
        sorted_indices = np.argsort(node_importance)[::-1]
        top_5_proteins = [protein_ids[idx] for idx in sorted_indices[:5]]
        results['all_top_proteins'].append([protein_ids[idx] for idx in sorted_indices])
        
        # Create label
        if isinstance(patient_ids, pd.Series):
            patient_id = patient_ids.iloc[i] if i < len(patient_ids) else f"unknown_{i}"
        else:
            patient_id = patient_ids[i] if i < len(patient_ids) else f"unknown_{i}"
        results['all_labels'].append(f'Top 5 for person {patient_id}: {", ".join(top_5_proteins)}')
        
        # Update cumulative importance
        for idx, importance in enumerate(node_importance):
            protein_id = protein_ids[idx]
            results['sum_node_importance_raw'][protein_id] += importance
            results['sum_node_importance_percent'][protein_id] += importance_percentages[idx]
            
            if importance > 0:
                results['positive_percent_by_protein'][protein_id] += (
                    (importance / total_positive) * 100 if total_positive != 0 else 0
                )
            elif importance < 0:
                results['negative_percent_by_protein'][protein_id] += (
                    (np.abs(importance) / total_negative) * 100 if total_negative != 0 else 0
                )
    
    def _create_importance_plots(self, results: Dict[str, Any], filename: str):
        """Create importance plots for the dataset."""
        self.plotter.plot_importance_scores(
            results['all_raw_importances'], 
            results['all_labels'], 
            f'{filename}_raw.png',
            'Sorted Raw Node Importance with Top 5 Protein IDs', 
            'Importance'
        )
        
        self.plotter.plot_importance_scores(
            results['all_percent_importances'], 
            results['all_labels'], 
            f'{filename}_percent.png',
            'Sorted Node Importance as Percentage of Total with Top 5 Protein IDs', 
            'Importance (%)'
        )

class DataExporter:
    """Handles exporting analysis results to various formats."""
    
    @staticmethod
    def export_to_csv(all_importances: List, protein_ids: np.ndarray, 
                     demographics: Tuple, model_name: str, 
                     importance_type: str = "percent") -> None:
        """Export importance data to CSV files."""
        df = pd.DataFrame(all_importances, 
                         index=demographics[3],  # total_did_labels
                         columns=protein_ids[:len(all_importances[0])])
        
        # Add demographic columns
        df.insert(0, "SEX", demographics[0])  # total_sex_labels
        df.insert(1, "AGE", demographics[2])   # total_age_labels
        df.insert(2, "Mutation", demographics[1])  # total_mutation_labels
        df.insert(3, "Gene.Dx", demographics[6])   # total_gene_labels
        
        filename = f"{importance_type}_importances_{model_name}.csv"
        df.to_csv(filename)
        print(f"Exported {filename}")



# Main orchestration functions
def run_explainer_train_and_test(checkpoint_path: str) -> Dict[str, Any]:
    """
    Main function to run explainer analysis on train and test datasets.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all analysis results
    """
    # Load model and config
    model, config = ModelLoader.load_checkpoint(checkpoint_path)
    config = ModelLoader.to_legacy_config(config)
    
    # Initialize components
    data_processor = DataProcessor(config)
    plotter = Plotter()
    
    # Load data
    train_dataset, test_dataset = data_processor.load_datasets()
    protein_ids, demographics = data_processor.get_demographics_and_protein_ids()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset.to(device)
    test_dataset.to(device)
    
    # Create explainer
    explainer_config = ExplainerConfig(top_k=config.num_nodes)
    analyzer = ExplainerAnalyzer(model, explainer_config, device, plotter)
    
    # Analyze datasets
    model_name = checkpoint_path.split("/")[-1]
    train_results = analyzer.analyze_dataset(train_dataset, protein_ids, demographics[4], f"{model_name}_train")
    test_results = analyzer.analyze_dataset(test_dataset, protein_ids, demographics[5], f"{model_name}_test")
    
    # Combine results
    combined_results = _combine_results(train_results, test_results)
    
    # Export data
    DataExporter.export_to_csv(combined_results['all_raw_importances'], protein_ids, 
                              demographics, model_name, "raw")
    DataExporter.export_to_csv(combined_results['all_percent_importances'], protein_ids, 
                              demographics, model_name, "percent")
    
    return {
        'combined_results': combined_results,
        'config': config,
        'protein_ids': protein_ids,
        'demographics': demographics
    }


def create_protein_plots(combined_results: Dict[str, Any], protein_ids: np.ndarray, 
                        config: Any, checkpoint_path: str) -> None:
    """
    Create comprehensive protein analysis plots.
    
    Parameters
    ----------
    combined_results : Dict[str, Any]
        Combined results from train and test analysis
    protein_ids : np.ndarray
        Array of protein IDs
    config : Any
        Configuration object
    checkpoint_path : str
        Path to the checkpoint (used for naming)
    """
    model_name = checkpoint_path.split("/")[-1]
    plotter = Plotter()
    
    # Calculate averages across all people
    avg_percent = _divide_dict_values(combined_results['combined_protein_count'], 
                                    combined_results['combined_sum_node_importance_percent'])
    avg_positive = _divide_dict_values(combined_results['combined_protein_count'], 
                                     combined_results['combined_positive_percent_by_protein'])
    avg_negative = _divide_dict_values(combined_results['combined_protein_count'], 
                                      combined_results['combined_negative_percent_by_protein'])
    avg_raw = _divide_dict_values(combined_results['combined_protein_count'], 
                                 combined_results['combined_sum_node_importance_raw'])
    
    # Create plots
    title_suffix = f"for {config.y_val} {config.sex} {config.mutation} {config.modality}"
    
    plotter.plot_bar_chart(avg_percent, f'Top Proteins Average Percentage Importance {title_suffix}',
                          'Protein ID', 'Importance Value (%)')
    
    plotter.plot_top_bar_chart(avg_positive, f'Top Positive Proteins Average Percentage Importance {title_suffix}',
                              'Protein ID', 'Importance Value (%)')
    
    plotter.plot_top_bar_chart(avg_negative, f'Top Negative Proteins Average Percentage Importance {title_suffix}',
                              'Protein ID', 'Importance Value (%)')
    
    plotter.plot_bar_chart(combined_results['combined_sum_node_importance_raw'], 
                          f'Sum of node importance for each protein {title_suffix}',
                          'Protein', 'Sum of node importance')
    
    plotter.plot_bar_chart(avg_raw, f'Top Proteins Average Importance Value {title_suffix}',
                          'Protein ID', 'Importance Value')


# Utility functions
def _combine_results(train_results: Dict[str, Any], test_results: Dict[str, Any]) -> Dict[str, Any]:
    """Combine train and test results."""
    combined = {}
    
    # Combine dictionaries
    for key in ['sum_node_importance_raw', 'sum_node_importance_percent', 
                'positive_percent_by_protein', 'negative_percent_by_protein']:
        combined[f'combined_{key}'] = {
            k: train_results[key].get(k, 0) + test_results[key].get(k, 0)
            for k in set(train_results[key]) | set(test_results[key])
        }
    
    # Combine lists and counters
    combined['combined_protein_count'] = train_results['protein_count'] + test_results['protein_count']
    print(combined['combined_protein_count'])
    combined['all_raw_importances'] = train_results['all_raw_importances'] + test_results['all_raw_importances']
    combined['all_percent_importances'] = train_results['all_percent_importances'] + test_results['all_percent_importances']
    combined['all_top_proteins'] = train_results['all_top_proteins'] + test_results['all_top_proteins']
    
    return combined


def _divide_dict_values(dict1: Dict, dict2: Dict) -> Dict:
    """Divide values of dict2 by dict1 for matching keys."""
    result = {}
    for key in dict1:
        if key in dict2 and isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
            if dict1[key] != 0:
                result[key] = dict2[key] / dict1[key]
            else:
                result[key] = None
        else:
            result[key] = None
    return result
    
