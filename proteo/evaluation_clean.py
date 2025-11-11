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
import captum.attr as C
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
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


IntegratedGradients = type(  
    "IntegratedGradients",
    (C.IntegratedGradients,),
    {
        "__init__": lambda self, forward_func, **kw: C.IntegratedGradients.__init__(
            self, forward_func, multiply_by_inputs=False, **kw
        )
    },
)

@dataclass
class ExplainerConfig:
    """Configuration for explainer analysis."""
    algorithm: str = 'IntegratedGradients'
    explanation_type: str = 'model'
    node_mask_type: str = 'attributes'
    edge_mask_type: Optional[str] = None
    threshold_type: str = 'topk'
    top_k: int = 7258
    mode: str = 'regression'
    task_level: str = 'graph'
    return_type: str = 'raw'

@dataclass
class PlotConfig:
    """Configuration for plotting parameters."""
    figure_size: Tuple[int, int] = (28, 10)
    dpi: int = 600
    bar_width: float = 0.6
    top_n: int = 25
    font_size: Dict[str, int] = None
    
    def __post_init__(self):
        if self.font_size is None:
            self.font_size = {
                'title': 48,
                'label': 44,
                'tick': 32,
                'annotation': 12
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
            "two_pass": p.get("two_pass", False),
            "two_pass_error_protein_file_name": p.get("two_pass_error_protein_file_name", None),
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
        # Set font to DejaVu Sans (similar to Arial, readily available on Linux)
        self._set_font_family()
    
    def _set_font_family(self):
        """Set font family with fallback options.
        
        Uses DejaVu Sans as primary font (very similar to Arial and readily available).
        Falls back to Arial if available, then to system sans-serif.
        """
        import matplotlib.font_manager as fm
        
        # Check available fonts (case-insensitive check)
        available_fonts = [f.name.lower() for f in fm.fontManager.ttflist]
        
        # Prefer DejaVu Sans (similar to Arial, typically available on Linux)
        # Then Arial if available, then system default
        if 'dejavu sans' in available_fonts:
            font_family = 'DejaVu Sans'
            font_list = ['DejaVu Sans', 'Arial', 'sans-serif']
        elif 'arial' in available_fonts:
            font_family = 'Arial'
            font_list = ['Arial', 'DejaVu Sans', 'sans-serif']
        else:
            # Use sans-serif as fallback (will use system default)
            font_family = 'sans-serif'
            font_list = ['sans-serif']
        
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.sans-serif'] = font_list
    
    def plot_importance_scores(self, explanations: List, labels: List, 
                              filename: str, title: str, ylabel: str, algo: Optional[str] = None) -> None:
        """Plot importance scores for multiple samples."""
        plt.figure()
        for i, importance in enumerate(explanations):
            plt.plot(sorted(importance, reverse=True), label=f'Person {i}')

        full_title = f"{title}  •  Algorithm: {algo}" if algo else title
        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                  fontsize='small', ncol=1)
        plt.xlabel('Protein')
        plt.ylabel(ylabel)
        plt.title(full_title)
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
                                   x_label, y_label, '#F5D6CC', 
                                   f"{filename}_highest.png" if filename else None)
        
        # Plot lowest
        self._plot_single_bar_chart(top_lowest, f"Top {top_n} Lowest - {title}", 
                                   x_label, y_label, '#C0D7DD', 
                                   f"{filename}_lowest.png" if filename else None)
    
    def _plot_single_bar_chart(self, data: Dict, title: str, x_label: str, 
                            y_label: str, color: str, filename: Optional[str] = None) -> None:
        truncated_labels = [key.split('|')[0] for key in data.keys()]
        y_values = list(data.values())
        x_positions = range(len(truncated_labels))
        
        self._set_font_family()
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax.bar(x_positions, y_values, color=color, width=self.config.bar_width, align='center', 
               edgecolor='black', linewidth=3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(truncated_labels, rotation=90, ha='center', fontsize=self.config.font_size['tick'])
        ax.set_xlabel(x_label, fontsize=self.config.font_size['label'])
        ax.set_ylabel(y_label, fontsize=self.config.font_size['label'])
        ax.set_title(title, fontsize=self.config.font_size['title'], pad=50)
        ax.tick_params(axis='y', labelsize=self.config.font_size['tick'])
        plt.tight_layout(pad=1.0)
        fig.subplots_adjust(bottom=0.25)
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=self.config.dpi)
        plt.show()
    
    def plot_top_bar_chart(self, protein_dict: Dict, title: str, x_label: str, 
                          y_label: str, filename: Optional[str] = None, top_n: int = None, 
                          color: Optional[str] = None) -> None:
        """Create a bar chart for top N highest values."""
        top_n = top_n or self.config.top_n
        
        # Determine color based on title/content - positive by default, negative if title contains "Negative"
        if color is None:
            if 'negative' in title.lower():
                color = '#C0D7DD'  # Negative color
            else:
                color = '#F5D6CC'  # Positive color
        
        # Sort and get top N highest
        sorted_items_desc = dict(sorted(protein_dict.items(), key=lambda item: item[1], reverse=True))
        top_highest = dict(list(sorted_items_desc.items())[:top_n])
        
        # Truncate protein names at first "|" for display
        truncated_labels = [key.split('|')[0] for key in top_highest.keys()]
        
        # Ensure font is set (handled in __init__, but ensure it's applied)
        self._set_font_family()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax.bar(truncated_labels, top_highest.values(), color=color, width=self.config.bar_width,
               edgecolor='black', linewidth=3)
        ax.set_xlabel(x_label, fontsize=self.config.font_size['label'])
        ax.set_ylabel(y_label, fontsize=self.config.font_size['label'])
        ax.set_title(f"Top {top_n} - {title}", fontsize=self.config.font_size['title'], pad=50)
        ax.tick_params(axis='x', rotation=90, labelsize=self.config.font_size['tick'])
        ax.tick_params(axis='y', labelsize=self.config.font_size['tick'])
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=self.config.font_size['tick'])
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{filename}_top_{top_n}.png", bbox_inches='tight', dpi=self.config.dpi)
        plt.show()

class ExplainerAnalyzer:
    """Handles explainer analysis and results processing."""
    
    def __init__(self, model, config: ExplainerConfig, device: str = 'cuda', plotter: Plotter = None, baseline_mean_tensor: torch.Tensor = None):
        self.model = model
        self.config = config
        self.device = device
        self.plotter = plotter or Plotter()
        self.explainer_baselines = baseline_mean_tensor
        self.explainer = self._create_explainer()
    
    def _create_explainer(self) -> Explainer:
        """Create explainer instance."""
        return Explainer(
            model=self.model.to(self.device),
            algorithm=CaptumExplainer(IntegratedGradients), #'IntegratedGradients', multiply_by_inputs = False), #, abs=False), # baselines=self.explainer_baselines),
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
            'Importance',
            algo=self.config.algorithm
        )
        
        self.plotter.plot_importance_scores(
            results['all_percent_importances'], 
            results['all_labels'], 
            f'{filename}_percent.png',
            'Sorted Node Importance as Percentage of Total with Top 5 Protein IDs', 
            'Importance (%)',
            algo=self.config.algorithm
        )

class DataExporter:
    """Handles exporting analysis results to various formats."""
    
    @staticmethod
    def export_to_csv(all_importances: List, protein_ids: np.ndarray, 
                     demographics: Tuple, model_name: str, 
                     importance_type: str = "percent",
                     algo: Optional[str] = None) -> None:
        """Export importance data to CSV files."""
        df = pd.DataFrame(all_importances, 
                         index=demographics[3],  # total_did_labels
                         columns=protein_ids[:len(all_importances[0])])
        
        # Add demographic columns
        df.insert(0, "SEX", demographics[0])  # total_sex_labels
        df.insert(1, "AGE", demographics[2])   # total_age_labels
        df.insert(2, "Mutation", demographics[1])  # total_mutation_labels
        df.insert(3, "Gene.Dx", demographics[6])   # total_gene_labels
        algo_tag = ""
        if algo:
            safe_algo = re.sub(r"[^A-Za-z0-9._-]+", "_", algo.strip())
            algo_tag = f"_{safe_algo}"
        filename = f"{importance_type}_importances_{model_name}{algo_tag}.csv"
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
    baseline_mean, features = data_processor.get_baseline_data()
    baseline_mean_tensor = torch.tensor(baseline_mean, dtype=torch.float32).to(device)
    baseline_mean_tensor = baseline_mean_tensor.unsqueeze(0).unsqueeze(2)
    analyzer = ExplainerAnalyzer(model, explainer_config, device, plotter, baseline_mean_tensor)
    
    # Analyze datasets
    model_name = checkpoint_path.split("/")[-1]
    train_results = analyzer.analyze_dataset(train_dataset, protein_ids, demographics[4], f"{model_name}_train")
    test_results = analyzer.analyze_dataset(test_dataset, protein_ids, demographics[5], f"{model_name}_test")
    
    # Combine results
    combined_results = _combine_results(train_results, test_results)
    
    # Export data
    DataExporter.export_to_csv(combined_results['all_raw_importances'], protein_ids, 
                              demographics, model_name, "raw", algo=explainer_config.algorithm)
    DataExporter.export_to_csv(combined_results['all_percent_importances'], protein_ids, 
                              demographics, model_name, "percent", algo=explainer_config.algorithm)
    
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
    
    # Get total number of people
    total_people = len(combined_results['all_raw_importances'])
    print(f"total_people = {total_people}")
    
    # Calculate averages by dividing by total number of people
    avg_percent = _divide_dict_by_scalar(combined_results['combined_sum_node_importance_percent'], total_people)
    avg_positive = _divide_dict_by_scalar(combined_results['combined_positive_percent_by_protein'], total_people)
    avg_negative = _divide_dict_by_scalar(combined_results['combined_negative_percent_by_protein'], total_people)
    avg_raw = _divide_dict_by_scalar(combined_results['combined_sum_node_importance_raw'], total_people)
    
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
    
    # Combine lists
    combined['all_raw_importances'] = train_results['all_raw_importances'] + test_results['all_raw_importances']
    combined['all_percent_importances'] = train_results['all_percent_importances'] + test_results['all_percent_importances']
    
    return combined


def _divide_dict_by_scalar(dictionary: Dict, scalar: Union[int, float]) -> Dict:
    """Divide all values in a dictionary by a scalar."""
    result = {}
    for key, value in dictionary.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and scalar != 0:
            result[key] = float(value) / scalar
        else:
            result[key] = None
    return result


def create_heatmap_from_csv(csv_file_path: str, output_filename: str = None, 
                           figsize: Tuple[int, int] = (20, 12), 
                           cmap: str = 'RdBu_r', top_n: int = 100) -> None:
    """
    Create a heatmap from a saved percent_importances CSV file showing top N most 
    positively important proteins on average per person.
    
    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing percent importances
    output_filename : str, optional
        Filename to save the heatmap. If None, displays the plot
    figsize : Tuple[int, int], optional
        Figure size for the heatmap (default: (20, 12))
    cmap : str, optional
        Colormap for the heatmap (default: 'RdBu_r')
    top_n : int, optional
        Number of top proteins to show (default: 100)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path, index_col=0)
    
    # Sort by Mutation column
    df_sorted = df.sort_values('Mutation')
    
    # Get columns that end with "|CSF"
    csf_columns = [col for col in df_sorted.columns if col.endswith('|CSF')]
    
    if not csf_columns:
        print("No columns ending with '|CSF' found in the CSV file.")
        return
    
    # Extract the CSF columns data
    csf_data = df_sorted[csf_columns]
    
    # Calculate average importance per protein across all samples
    protein_averages = csf_data.mean()
    
    # Get top N most positively important proteins (highest average values)
    top_proteins = protein_averages.nlargest(top_n).index.tolist()
    
    # Filter data to only include top proteins
    top_csf_data = csf_data[top_proteins]
    
    # Create the heatmap
    plt.figure(figsize=figsize, dpi=300)
    
    # Create heatmap with seaborn
    sns.heatmap(top_csf_data.T,  # Transpose so proteins are on y-axis, samples on x-axis
                cmap=cmap,
                center=0,  # Center the colormap at 0
                cbar_kws={'label': 'Importance Score'},
                xticklabels=False,  # Hide x-axis labels for readability
                yticklabels=True)
    
    # Customize the plot
    plt.title(f'Top {top_n} Most Positively Important CSF Proteins\nSorted by Mutation\n({len(top_proteins)} proteins, {len(df_sorted)} samples)', 
              fontsize=16, pad=20)
    plt.xlabel('Samples (sorted by Mutation)', fontsize=14)
    plt.ylabel('CSF Proteins (ranked by average importance)', fontsize=14)
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=8)
    
    # Add mutation labels as text annotations on x-axis
    mutation_counts = df_sorted['Mutation'].value_counts().sort_index()
    mutation_positions = []
    current_pos = 0
    
    for mutation, count in mutation_counts.items():
        mutation_positions.append(current_pos + count // 2)
        current_pos += count
    
    # Add mutation labels
    for i, (mutation, pos) in enumerate(zip(mutation_counts.index, mutation_positions)):
        plt.text(pos, -0.5, mutation, ha='center', va='top', 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved as {output_filename}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\nHeatmap Summary:")
    print(f"- Total samples: {len(df_sorted)}")
    print(f"- Total CSF proteins available: {len(csf_columns)}")
    print(f"- Top {top_n} proteins shown: {len(top_proteins)}")
    print(f"- Mutations: {', '.join(mutation_counts.index)}")
    print(f"- Mutation distribution: {dict(mutation_counts)}")
    
    # Print top 10 proteins by average importance
    print(f"\nTop 10 proteins by average importance:")
    for i, (protein, avg_importance) in enumerate(protein_averages.nlargest(10).items(), 1):
        print(f"{i:2d}. {protein}: {avg_importance:.6f}")
    
