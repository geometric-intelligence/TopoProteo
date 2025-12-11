import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import zscore, pearsonr
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import json

from topobench.data.datasets.ftd_dataset import (
    reverse_log_transform,
    Y_VALS_TO_NORMALIZE,
    FTDDataset
)
from topobench.data.utils.utils import construct_datasets
from proteo.evaluation_clean import ModelLoader, DataProcessor
from omegaconf import OmegaConf
from hydra.utils import instantiate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_checkpoint_for_inference(ckpt_path: str, map_location: str = "cpu"):
    """
    Load model checkpoint without modifying forward method (for regular inference).
    
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
    
    # Don't modify forward method - use original for batch inference
    model.eval()
    return model, cfg


def load_stats_from_json(config):
    """Load mean and std from JSON file for reverse transformation."""
    root = config.data_dir
    dataset = FTDDataset(root, config, "train")
    
    # Check if batch_normalization is enabled
    batch_norm_str = "_batch_normalization" if getattr(config, 'batch_normalization', False) else ""
    
    if config.kfold:
        mean_std_file_name = (
            f"{dataset.experiment_id}_train_random_state_{config.random_state}_"
            f"{config.num_folds}fold_{config.fold}{batch_norm_str}.json"
        )
    else:
        mean_std_file_name = (
            f"{dataset.experiment_id}_train_random_state_{config.random_state}{batch_norm_str}.json"
        )
    
    mean_std_file_path = os.path.join(dataset.processed_dir, mean_std_file_name)
    
    if not os.path.exists(mean_std_file_path):
        print(f"Warning: Stats file not found at {mean_std_file_path}")
        print("Predictions will be in normalized units.")
        return None, None
    
    with open(mean_std_file_path, 'r') as f:
        print(f"Loading stats from: {mean_std_file_path}")
        content = f.read()
        # Handle both JSON format and text format
        if content.strip().startswith('{'):
            stats = json.loads(content)
            mean = stats.get('mean')
            std = stats.get('std')
        else:
            # Text format: "mean: X\nstd: Y"
            mean = float(content.split("mean: ")[1].split("\n")[0])
            std = float(content.split("std: ")[1].split("\n")[0])
    
    return mean, std


def get_predictions(model, dataset, split_name, device='cuda'):
    """Get predictions and targets for a dataset."""
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []
    targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Ensure batch has the expected structure for the model
            # The model may expect x_0 and batch_0 instead of x and batch
            if 'x' in batch and 'x_0' not in batch:
                batch.x_0 = batch.x
            
            # Ensure batch_0 exists (required by model)
            # PyTorch Geometric DataLoader creates 'batch' attribute automatically
            # The model expects 'batch_0' instead
            if 'batch' in batch:
                if 'batch_0' not in batch:
                    batch.batch_0 = batch.batch
            else:
                # If batch doesn't exist, create batch_0
                # This shouldn't happen with DataLoader, but handle it just in case
                num_nodes = batch.x.size(0) if 'x' in batch else (batch.x_0.size(0) if 'x_0' in batch else 1)
                device_tensor = batch.x.device if 'x' in batch else (batch.x_0.device if 'x_0' in batch else device)
                batch.batch_0 = torch.zeros(num_nodes, dtype=torch.int64, device=device_tensor)
            
            # Get prediction
            pred = model(batch)
            
            # Handle different output shapes (dict, tuple, or tensor)
            if isinstance(pred, dict):
                pred = pred.get("logits", next(v for v in pred.values() if torch.is_tensor(v)))
            elif isinstance(pred, (tuple, list)):
                pred = pred[0]
            
            if pred.dim() > 1:
                pred = pred.squeeze()
            
            target = batch.y
            if target.dim() > 1:
                target = target.squeeze()
            
            predictions.append(pred.cpu())
            targets.append(target.cpu())
    
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    return predictions, targets


def calculate_metrics(predictions, targets):
    """Calculate regression metrics."""
    predictions_np = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    targets_np = targets.numpy() if isinstance(targets, torch.Tensor) else targets
    
    # Ensure they're 1D
    predictions_np = predictions_np.flatten()
    targets_np = targets_np.flatten()
    
    # Convert to tensors for MSE calculation
    pred_tensor = torch.tensor(predictions_np, dtype=torch.float32)
    targ_tensor = torch.tensor(targets_np, dtype=torch.float32)
    
    mse = F.mse_loss(pred_tensor, targ_tensor).item()
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_np - targets_np))
    r2 = 1 - (np.sum((targets_np - predictions_np) ** 2) / 
              np.sum((targets_np - np.mean(targets_np)) ** 2))
    pearson_r, pearson_p = pearsonr(targets_np, predictions_np)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }


def full_load_and_run_and_convert(checkpoint_path, device='cuda'):
    """
    Load checkpoint, run model, and convert predictions to original units.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file
    device : str
        Device to run inference on ('cuda' or 'cpu')
    
    Returns
    -------
    dict
        Dictionary containing predictions, targets, and metrics for train/val/test
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load model and config without modifying forward method (for batch inference)
    model, cfg = load_checkpoint_for_inference(checkpoint_path, map_location=device)
    config = ModelLoader.to_legacy_config(cfg)
    
    print(f"Config: y_val={config.y_val}, modality={config.modality}, "
          f"mutation={config.mutation}, sex={config.sex}")
    
    # Load datasets
    print("Loading datasets...")
    data_processor = DataProcessor(config)
    train_dataset, val_dataset, test_dataset = data_processor.load_datasets()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Get predictions for all splits
    print("\nGetting train predictions...")
    train_preds, train_targets = get_predictions(model, train_dataset, "Train", device)
    
    print("Getting val predictions...")
    val_preds, val_targets = get_predictions(model, val_dataset, "Val", device)
    
    print("Getting test predictions...")
    test_preds, test_targets = get_predictions(model, test_dataset, "Test", device)
    
    # Calculate metrics in normalized units
    print("\n=== Metrics in Normalized Units ===")
    train_metrics_norm = calculate_metrics(train_preds, train_targets)
    val_metrics_norm = calculate_metrics(val_preds, val_targets)
    test_metrics_norm = calculate_metrics(test_preds, test_targets)
    
    print(f"\nTrain (normalized):")
    print(f"  MSE: {train_metrics_norm['mse']:.4f}, RMSE: {train_metrics_norm['rmse']:.4f}, "
          f"MAE: {train_metrics_norm['mae']:.4f}, R²: {train_metrics_norm['r2']:.4f}, "
          f"Pearson r: {train_metrics_norm['pearson_r']:.4f}")
    
    print(f"\nVal (normalized):")
    print(f"  MSE: {val_metrics_norm['mse']:.4f}, RMSE: {val_metrics_norm['rmse']:.4f}, "
          f"MAE: {val_metrics_norm['mae']:.4f}, R²: {val_metrics_norm['r2']:.4f}, "
          f"Pearson r: {val_metrics_norm['pearson_r']:.4f}")
    
    print(f"\nTest (normalized):")
    print(f"  MSE: {test_metrics_norm['mse']:.4f}, RMSE: {test_metrics_norm['rmse']:.4f}, "
          f"MAE: {test_metrics_norm['mae']:.4f}, R²: {test_metrics_norm['r2']:.4f}, "
          f"Pearson r: {test_metrics_norm['pearson_r']:.4f}")
    
    # Initialize metrics_orig variables
    train_metrics_orig = None
    val_metrics_orig = None
    test_metrics_orig = None
    
    # Convert to original units if normalization was used
    if config.y_val in Y_VALS_TO_NORMALIZE:
        mean, std = load_stats_from_json(config)
        
        if mean is not None and std is not None:
            print(f"\nConverting to original units (mean={mean:.4f}, std={std:.4f})...")
            
            train_preds_orig = reverse_log_transform(train_preds, mean, std)
            train_targets_orig = reverse_log_transform(train_targets, mean, std)
            val_preds_orig = reverse_log_transform(val_preds, mean, std)
            val_targets_orig = reverse_log_transform(val_targets, mean, std)
            test_preds_orig = reverse_log_transform(test_preds, mean, std)
            test_targets_orig = reverse_log_transform(test_targets, mean, std)
            
            # Calculate metrics in original units
            print("\n=== Metrics in Original Units ===")
            train_metrics_orig = calculate_metrics(train_preds_orig, train_targets_orig)
            val_metrics_orig = calculate_metrics(val_preds_orig, val_targets_orig)
            test_metrics_orig = calculate_metrics(test_preds_orig, test_targets_orig)
            
            print(f"\nTrain (original units):")
            print(f"  MSE: {train_metrics_orig['mse']:.4f}, RMSE: {train_metrics_orig['rmse']:.4f}, "
                  f"MAE: {train_metrics_orig['mae']:.4f}, R²: {train_metrics_orig['r2']:.4f}, "
                  f"Pearson r: {train_metrics_orig['pearson_r']:.4f}")
            
            print(f"\nVal (original units):")
            print(f"  MSE: {val_metrics_orig['mse']:.4f}, RMSE: {val_metrics_orig['rmse']:.4f}, "
                  f"MAE: {val_metrics_orig['mae']:.4f}, R²: {val_metrics_orig['r2']:.4f}, "
                  f"Pearson r: {val_metrics_orig['pearson_r']:.4f}")
            
            print(f"\nTest (original units):")
            print(f"  MSE: {test_metrics_orig['mse']:.4f}, RMSE: {test_metrics_orig['rmse']:.4f}, "
                  f"MAE: {test_metrics_orig['mae']:.4f}, R²: {test_metrics_orig['r2']:.4f}, "
                  f"Pearson r: {test_metrics_orig['pearson_r']:.4f}")
            
            # Use original units for plotting
            train_preds_plot = train_preds_orig
            train_targets_plot = train_targets_orig
            val_preds_plot = val_preds_orig
            val_targets_plot = val_targets_orig
            test_preds_plot = test_preds_orig
            test_targets_plot = test_targets_orig
        else:
            print("\nWarning: Could not load mean/std. Using normalized units for plots.")
            train_preds_plot = train_preds
            train_targets_plot = train_targets
            val_preds_plot = val_preds
            val_targets_plot = val_targets
            test_preds_plot = test_preds
            test_targets_plot = test_targets
    else:
        print("\nTarget not normalized, using predictions as-is.")
        train_preds_plot = train_preds
        train_targets_plot = train_targets
        val_preds_plot = val_preds
        val_targets_plot = val_targets
        test_preds_plot = test_preds
        test_targets_plot = test_targets
    
    # Convert to numpy for plotting
    train_preds_np = train_preds_plot.cpu().numpy() if isinstance(train_preds_plot, torch.Tensor) else train_preds_plot
    train_targets_np = train_targets_plot.cpu().numpy() if isinstance(train_targets_plot, torch.Tensor) else train_targets_plot
    val_preds_np = val_preds_plot.cpu().numpy() if isinstance(val_preds_plot, torch.Tensor) else val_preds_plot
    val_targets_np = val_targets_plot.cpu().numpy() if isinstance(val_targets_plot, torch.Tensor) else val_targets_plot
    test_preds_np = test_preds_plot.cpu().numpy() if isinstance(test_preds_plot, torch.Tensor) else test_preds_plot
    test_targets_np = test_targets_plot.cpu().numpy() if isinstance(test_targets_plot, torch.Tensor) else test_targets_plot
    
    # Flatten arrays
    train_preds_np = train_preds_np.flatten()
    train_targets_np = train_targets_np.flatten()
    val_preds_np = val_preds_np.flatten()
    val_targets_np = val_targets_np.flatten()
    test_preds_np = test_preds_np.flatten()
    test_targets_np = test_targets_np.flatten()
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Train
    axes[0].scatter(train_targets_np, train_preds_np, alpha=0.6, s=30, 
                   edgecolors='black', linewidths=0.5, color='blue')
    min_val = min(train_targets_np.min(), train_preds_np.min())
    max_val = max(train_targets_np.max(), train_preds_np.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('True Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title('Train Set', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Val
    axes[1].scatter(val_targets_np, val_preds_np, alpha=0.6, s=30, 
                   edgecolors='black', linewidths=0.5, color='orange')
    min_val = min(val_targets_np.min(), val_preds_np.min())
    max_val = max(val_targets_np.max(), val_preds_np.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[1].set_xlabel('True Values', fontsize=12)
    axes[1].set_ylabel('Predicted Values', fontsize=12)
    axes[1].set_title('Validation Set', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot Test
    axes[2].scatter(test_targets_np, test_preds_np, alpha=0.6, s=30, 
                   edgecolors='black', linewidths=0.5, color='green')
    min_val = min(test_targets_np.min(), test_preds_np.min())
    max_val = max(test_targets_np.max(), test_preds_np.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[2].set_xlabel('True Values', fontsize=12)
    axes[2].set_ylabel('Predicted Values', fontsize=12)
    axes[2].set_title('Test Set', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    checkpoint_name = os.path.basename(checkpoint_path).replace('.ckpt', '')
    save_path = f'true_vs_predicted_{checkpoint_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    # Return results
    results = {
        'train': {
            'predictions': train_preds_plot,
            'targets': train_targets_plot,
            'metrics_norm': train_metrics_norm,
            'metrics_orig': train_metrics_orig
        },
        'val': {
            'predictions': val_preds_plot,
            'targets': val_targets_plot,
            'metrics_norm': val_metrics_norm,
            'metrics_orig': val_metrics_orig
        },
        'test': {
            'predictions': test_preds_plot,
            'targets': test_targets_plot,
            'metrics_norm': test_metrics_norm,
            'metrics_orig': test_metrics_orig
        }
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "/scratch/lcornelis/outputs/checkpoints/epoch_024-v1210.ckpt"
    results = full_load_and_run_and_convert(checkpoint_path, device=device)
