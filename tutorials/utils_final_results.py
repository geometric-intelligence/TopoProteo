import networkx as nx
import numpy as np
import csv
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra  # Import GlobalHydra explicitly
from topobench.utils.config_resolvers import (
    get_default_metrics,
    get_default_transform,
    get_flattened_feature_matrix_dim,
    get_gatv4_output_dim,
    get_monitor_metric,
    get_monitor_mode,
    get_required_lifting,
    infer_in_channels,
    infer_num_cell_dimensions,
)
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver(
    "get_default_metrics", get_default_metrics, replace=True
)
OmegaConf.register_new_resolver(
    "get_default_transform", get_default_transform, replace=True
)
OmegaConf.register_new_resolver(
    "get_flattened_feature_matrix_dim", get_flattened_feature_matrix_dim, replace=True
)
OmegaConf.register_new_resolver(
    "get_gatv4_output_dim", get_gatv4_output_dim, replace=True
)
OmegaConf.register_new_resolver(
    "get_required_lifting", get_required_lifting, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_metric", get_monitor_metric, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_mode", get_monitor_mode, replace=True
)
OmegaConf.register_new_resolver(
    "infer_in_channels", infer_in_channels, replace=True
)
OmegaConf.register_new_resolver(
    "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
)
OmegaConf.register_new_resolver(
    "parameter_multiplication", lambda x, y: int(int(x) * int(y)), replace=True
)

# Clear GlobalHydra instance if already initialized
if GlobalHydra().is_initialized():
    GlobalHydra().clear()

initialize(config_path="../configs", job_name="job")

def get_config_and_checkpoint(model, adj_metric, df):
    """
    Get the configuration for the model with specified overrides.
    
    Parameters:
    -----------
    model : str
        The model name.
    adj_metric : str
        The adjacency metric.
    df : pd.DataFrame
        DataFrame containing results, used to extract the overrides.

    Returns:
    --------
    DictConfig
        The resolved configuration.
    str
        The path to the checkpoint file.
    """
    run_info = df[df["model"]==model][df["dataset"]==adj_metric]
    
    if model == "mlp":
        model_overrides = []
    elif model == "gcn":
        model_overrides = [
            f"model.feature_encoder.out_channels={run_info['model.feature_encoder.out_channels'].iloc[0]}",
            f"model.backbone.dropout={run_info['model.backbone.dropout'].iloc[0]}",
            f"model.backbone.act={run_info['model.backbone.act'].iloc[0]}",
            f"model.backbone.num_layers={int(run_info['model.backbone.num_layers'].iloc[0])}"
        ]
    elif model == "gatv4":
        model_overrides = [
            f"model.feature_encoder.out_channels={run_info['model.feature_encoder.out_channels'].iloc[0]}",
            f"model.backbone.dropout={run_info['model.backbone.dropout'].iloc[0]}",
            f"model.backbone.act={run_info['model.backbone.act'].iloc[0]}",
            f"model.backbone.hidden_channels={run_info['model.backbone.hidden_channels'].iloc[0]}",
            f"model.backbone.heads={run_info['model.backbone.heads'].iloc[0]}".replace(" ", ""),
            f"model.backbone.weight_initializer={run_info['model.backbone.weight_initializer'].iloc[0]}"
        ]

    cfg = compose(
        config_name="run.yaml",
        overrides=[
            f"model=graph/{model}",
            "dataset=graph/FTD",
            f"dataset.loader.parameters.adj_metric={adj_metric}",
            f"dataset.loader.parameters.adj_thresh={run_info['dataset.loader.parameters.adj_thresh'].iloc[0]}",
            f"dataset.loader.parameters.kfold={run_info['dataset.loader.parameters.kfold'].iloc[0]}",
            f"dataset.loader.parameters.num_folds={run_info['dataset.loader.parameters.num_folds'].iloc[0]}",
            f"dataset.loader.parameters.fold={run_info['fold'].iloc[0]}",
            f"dataset.dataloader_params.batch_size={run_info['dataset.dataloader_params.batch_size'].iloc[0]}",
            f"model.readout.graph_encoder_dim={run_info['model.readout.graph_encoder_dim'].iloc[0]}".replace(" ", ""),
            f"model.readout.feature_encoder_dim={run_info['model.readout.feature_encoder_dim'].iloc[0]}",
            f"model.readout.fc_dim={run_info['model.readout.fc_dim'].iloc[0]}".replace(" ", ""),
            f"model.readout.fc_dropout={run_info['model.readout.fc_dropout'].iloc[0]}",
            f"model.readout.fc_act={run_info['model.readout.fc_act'].iloc[0]}",
            f"optimizer.parameters.lr={run_info['optimizer.parameters.lr'].iloc[0]}",
            f"dataset.split_params.data_seed={run_info['dataset.split_params.data_seed'].iloc[0]}",
            f"trainer.max_epochs={run_info['trainer.max_epochs'].iloc[0]}",
            f"trainer.min_epochs={run_info['trainer.min_epochs'].iloc[0]}",
            "trainer.check_val_every_n_epoch=1",
            "trainer.devices=[7]",
            "paths.work_dir=/tmp",
        ] + model_overrides,
        return_hydra_config=False
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    
    checkpoint = run_info['checkpoint'].iloc[0]
    
    return cfg, checkpoint

def load_model_checkpoint(cfg, checkpoint_dir):
    """
    Load the model checkpoint.
    
    Parameters:
    -----------
    cfg : DictConfig
        The configuration for the model.
    checkpoint_dir : str
        The path to the checkpoint directory.

    Returns:
    --------
    torch.nn.Module
        The loaded model.
    """
    model = instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model

def load_dataset(adj_metric, adj_thresh, kfold=True, num_folds=5, fold=0, y_val="nfl", two_pass=False, num_nodes=7258):
    """
    Load the FTD dataset with specified params.
    
    Parameters:
    -----------
    adj_metric : str
        The adjacency metric to use.
    adj_thresh : float
        The adjacency threshold.
    kfold : bool, optional
        Whether to use k-fold cross-validation (default is True).
    num_folds : int, optional
        The number of folds for cross-validation (default is 5).
    fold : int, optional
        The specific fold to use (default is 0).
    """
    cfg = compose(
        config_name="run.yaml",
        overrides=[
            "model=graph/gcn",
            "dataset=graph/FTD",
            f"dataset.loader.parameters.adj_metric={adj_metric}",
            f"dataset.loader.parameters.adj_thresh={adj_thresh}",
            f"dataset.loader.parameters.kfold={kfold}",
            f"dataset.loader.parameters.num_folds={num_folds}",
            f"dataset.loader.parameters.fold={fold}",
            f"dataset.loader.parameters.y_val={y_val}",
            f"dataset.loader.parameters.two_pass={two_pass}",
            f"dataset.loader.parameters.num_nodes={num_nodes}",
        ],
        return_hydra_config=True
    )
    loader = instantiate(cfg.dataset.loader)
    return loader.get_splits()