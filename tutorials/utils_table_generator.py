import os 
import wandb
import numpy as np
import pandas as pd
import itertools

def flatten_config(d, parent_key="", sep="."):
    items = []
    for k,v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_mean_std(cfg):
    """
    Given a configuration dictionary `cfg`, this function retrieves the mean and standard deviation
    of the training set used to normalize the training and validation labels.

    Parameters:
    -----------
    cfg : dict
        Configuration dictionary containing dataset parameters.
        
    Returns:
    --------
    tuple
        A tuple containing the mean and standard deviation of the training set.
    """
    processed_dir = "/scratch/lcornelis/data/data_louisa/FTD/processed"
    # split = cfg.get("dataset.loader.parameters.split",None)
    kfold = cfg.get("dataset.loader.parameters.kfold",None)
    num_folds = cfg.get("dataset.loader.parameters.num_folds",None)
    fold = cfg.get("dataset.loader.parameters.fold",None)
    adj_metric = cfg.get("dataset.loader.parameters.adj_metric",None)
    adj_thresh = cfg.get("dataset.loader.parameters.adj_thresh",None)
    adj_str = "adj_thresh_1.0" if adj_thresh==1 else f"adj_thresh_{adj_thresh}"
    y_val = cfg.get("dataset.loader.parameters.y_val",None)
    y_val_str = f"y_val_{y_val}" if y_val is not None else "y_val_None"
    num_nodes = cfg.get("dataset.loader.parameters.num_nodes",None)
    num_nodes_str = f"num_nodes_{num_nodes}" if num_nodes is not None else "num_nodes_None"
    mutation = cfg.get("dataset.loader.parameters.mutation",None)
    mutation_str = f"mutation_{','.join(mutation)}" if mutation is not None else "mutation_None"
    modality = cfg.get("dataset.loader.parameters.modality",None)
    modality_str = f"{modality}" if modality is not None else "None"
    sex = cfg.get("dataset.loader.parameters.sex",None)
    sex_str = f"sex_{','.join(sex)}"
    # y_val_str = f"y_val_{cfg.get("dataset.loader.parameters.y_val",None)}"
    # num_nodes_str = f"num_nodes_{cfg.get("dataset.loader.parameters.num_nodes",None)}"
    # mutation_str = f"mutation_{','.join(cfg.get("dataset.loader.parameters.mutation",None))}"
    # modality_str = f"{cfg.get("dataset.loader.parameters.modality",None)}"
    # sex_str = f"sex_{','.join(cfg.get("dataset.loader.parameters.sex",None))}"
    random_state = cfg.get("dataset.loader.parameters.random_state",None)

    experiment_id = f"FTD_{y_val_str}_{adj_metric}_{adj_str}_{num_nodes_str}_{mutation_str}_{modality_str}_{sex_str}"
    if kfold:
        file_name = f"{experiment_id}_train_random_state_{random_state}_{num_folds}fold_{fold}.json"
    else:
        file_name = (
            f"{experiment_id}_train_random_state_{random_state}.json"
        )
    file_path = os.path.join(processed_dir, file_name)

    mean, std = None, None
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("mean:"):
                mean = float(line.split("mean:")[1].strip())
            elif line.startswith("std:"):
                std = float(line.split("std:")[1].strip())
    return mean, std


def load_results_dataframe(wandb_username, wandb_project, original_units=True, metric="mse", csv_filename="proteo_results.csv", force_load=False, save_csv=True, filters={}):
    """
    Load results from W&B and return a DataFrame with the relevant metrics.

    Parameters:
    -----------
    wandb_username : str
        W&B username.
    wandb_project : str
        W&B project name.
    original_units : bool
        Whether to convert metrics back to original units.
    metric : str, optional
        The metric to extract from W&B runs (default is "mse").
    csv_filename : str, optional
        If provided, load results from this CSV file instead of W&B.
    force_load : bool, optional
        Whether to force load results from W&B even if the CSV file exists (default is False).
    save_csv : bool
        Whether to save the results to a CSV file (default is True).
    filters : dict, optional
        Filters to apply when fetching runs from W&B.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the results.
    """
    if os.path.exists(csv_filename) and not force_load:
        df = pd.read_csv(csv_filename)
    else:
        # ── A) CONFIGURE YOUR W&B ACCESS ────────────────────────────────────────────
        api = wandb.Api()
        runs = api.runs(f"{wandb_username}/{wandb_project}", filters=filters)
        print(f"▶ Number of runs fetched from W&B: {len(runs)}")

        # ── B) BUILD THE RAW DATAFRAME ────────────────────────────────────────────────
        records = []
        for run in runs:
            cfg = run.config.copy() or {}
            
            cfg = flatten_config(cfg)
            # Attempt to extract metrics—skip if missing
            if ("test/"+metric not in run.summary) or ("test/"+metric not in run.summary):
                continue
            # "dataset.loader.parameters.data_name", "model.backbone._target_", "dataset.split_params.data_seed"
            # Get mean and std used for normalization
            mean, std = get_mean_std(cfg)
            assert mean is not None, f"Mean is None for run {run.id}"
            assert std is not None, f"Std is None for run {run.id}"
            
            if original_units:
                # Convert validation and test metrics back to original units
                if metric == "mse":
                    val_mae  = np.sqrt(run.summary["test/"+metric]) * std 
                    test_mae = np.sqrt(run.summary["test/"+metric]) * std
                else:
                    # For other metrics, just multiply by std
                    val_mae = run.summary["test/"+metric] * std 
                    test_mae = run.summary["test/"+metric] * std 
            else:
                # Keep metrics in normalized units
                val_mae  = run.summary["test/"+metric]
                test_mae = run.summary["test/"+metric]
            dataset  = cfg.get("dataset.loader.parameters.adj_metric", None)
            model    = cfg.get("model.model_name",   None)
            fold     = cfg.get("dataset.loader.parameters.fold",    None)
            checkpoint = run.summary.get("checkpoint",None)
            best_epoch_checkpoint = run.summary.get("best_epoch/checkpoint",None)

            # If any of these is None, we might want to skip as well:
            if (dataset is None) or (model is None) or (fold is None):
                continue

            # Collect any other hyperparams (besides dataset/model/fold)
            hyperparams = {k: v for k, v in cfg.items() 
                        if k not in ["dataset", "model", "fold", "dataset.loader.parameters.adj_metric", "model.model_name", "dataset.loader.parameters.fold"]}

            row = {
                "dataset":  dataset,
                "model":    model,
                "fold": fold,
                "val_mae":  val_mae,
                "test_mae": test_mae,
                "checkpoint": checkpoint,
                "best_epoch/checkpoint": best_epoch_checkpoint,
            }
            row.update(hyperparams)
            records.append(row)

        df = pd.DataFrame(records)
        if save_csv:
            df.to_csv(csv_filename, index=False)
    print("▶ After building df, df.shape =", df.shape)
    return df

def filter_dataframe(df, columns):
    """
    Filter the DataFrame to only include specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to filter.
    columns : list of str
        List of column names to keep in the DataFrame.

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only the specified columns.
    """
    return df[columns] if all(col in df.columns for col in columns) else df


def generate_table(df, save_csv=False, csv_filename="proteo_results.csv"):
    """
    Group the DataFrame by dataset, model and fold, and compute the mean and standard deviation of the metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results.
    save_csv : bool
        Whether to save the grouped results to a CSV file.
    csv_filename : str, optional
        If provided, save the grouped results to this CSV filename (adding "grouped_").
    """
    # Filter to only keep runs with adj_thresh = 0.5
    # adj_thresh_col = "dataset.loader.parameters.adj_thresh"
    # if adj_thresh_col in df.columns:
    #     df = df[df[adj_thresh_col] == 0.3].copy()
    #     print(f"▶ Filtered to adj_thresh=0.5, df.shape = {df.shape}")
    # else:
    #     print(f"⚠ Warning: Column '{adj_thresh_col}' not found in datafrasme")
    
    # Remove checkpoint column, not wanted here
    df = df.drop(columns=['checkpoint'])

    # ── C) ENSURE ALL CELLS ARE HASHABLE FIRST ────────────────────────────────────
    def ensure_hashable(x):
        try:
            hash(x)
            return x
        except TypeError:
            if isinstance(x, np.ndarray):
                return tuple(x.tolist())
            elif isinstance(x, list):
                return tuple(x)
            elif isinstance(x, dict):
                return tuple(sorted(x.items()))
            else:
                return str(x)

    # Apply to all object columns to make them hashable before nunique()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(ensure_hashable)

    # Remove columns with only one unique value, except those in exclude_cols
    exclude_cols = ["dataset", "model", "fold", "val_mae", "test_mae"]
    nunique = df.nunique()
    keep_cols = [col for col in df.columns if (nunique[col] > 1) or (col in exclude_cols)]
    df = df[keep_cols]

    # ── C) DETERMINE HYPERPARAM COLUMNS ──────────────────────────────────────────
    static_cols    = {"dataset", "model", "fold", "val_mae", "test_mae"}
    hyperparam_cols = sorted(list(set(df.columns) - static_cols))
    print(f"▶ hyperparam_cols = {hyperparam_cols}")

    # If df is empty or columns are missing, fix those first
    if df.empty:
        raise RuntimeError("DataFrame `df` is empty. Check that runs actually contained "
                        "`dataset`, `model`, `fold`, `val/accuracy`, `test/accuracy`.")

        # Convert tuples to strings for display purposes (hashable conversion done earlier)
    for col in ["dataset", "model"] + hyperparam_cols:
        if col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (tuple, list))).any():
                df[col] = df[col].apply(str)

    # ── E) GROUP BY (dataset, model, hyperparams) ─────────────────────────────────
    group_cols = ["dataset", "model"] + hyperparam_cols
    print(f"▶ group_cols = {group_cols}")

    grouped = (
        df
        .groupby(group_cols, dropna=False)
        .agg(
            mean_val_mae = ("val_mae", "mean"),
            std_val_mae = ("val_mae", "std"),
            n_folds      = ("fold", "count")
        )
        .reset_index()
    )
    # Remove grouped rows with n_folds < 3 for models 'mlp' and 'gcn'
    mask = ~(grouped["n_folds"] < 5)
    grouped = grouped[mask].reset_index(drop=True)
    print("▶ After grouping, grouped.shape =", grouped.shape)

    if grouped.empty:
        raise RuntimeError("`grouped` is empty. Either hyperparam_cols are wrong or "
                        "no rows survived the groupby.")
    
    if save_csv:
        # Save the grouped DataFrame to a CSV file
        grouped_csv_filename = f"grouped_{csv_filename}"
        grouped.to_csv(grouped_csv_filename, index=False)
        print(f"▶ Grouped results saved to {grouped_csv_filename}")
        
    
    # ── F) PICK BEST CONFIGS PER (dataset, model) ─────────────────────────────────
    grouped_sorted = grouped.sort_values(
        by=["dataset", "model", "mean_val_mae"], 
        ascending=[True, True, True]
    )
    best_configs = (
        grouped_sorted
        .groupby(["dataset", "model"], as_index=False)
        .head(n=1)
    )
    print("▶ best_configs.shape =", best_configs.shape)

    # Build a “config_key” column in both DataFrames for merging
    def columns_config_key(row, cols):
        return tuple(c for c in cols if row[c] is not None)

    def make_config_key(row, cols):
        return tuple(row[c] for c in cols)

    best_configs = best_configs.fillna(np.nan)  # Ensure no NaN values
    grouped_cols = df.apply(lambda r: columns_config_key(r, group_cols[:]), axis=1)[0]  # same as group_cols

    best_configs["config_key"] = best_configs.apply(lambda r: make_config_key(r, grouped_cols), axis=1)
    df["config_key"]          = df.apply(lambda r: make_config_key(r, grouped_cols), axis=1)
    
    # ── G) FILTER df BY best_configs AND COMPUTE TEST STATS ────────────────────────
    chosen_keys = [elem for elem in best_configs["config_key"]]
    df_best_runs = df[df["config_key"].isin(chosen_keys)].copy()
    print("▶ df_best_runs.shape (should be #datasets × #models × #folds) =", df_best_runs.shape)

    if df_best_runs.empty:
        raise RuntimeError("`df_best_runs` is empty. That means none of your runs "
                        "matched the chosen best_configs.\n"
                        "Check if the config_key logic is correct.")

    # Group only by dataset and model to get one result per combination
    summary_group_cols = ["dataset", "model"]

    summary = (
        df_best_runs
        .groupby(summary_group_cols)
        .agg(
            mean_test_mae = ("test_mae", "mean"),
            std_test_mae  = ("test_mae", "std"),
        )
        .reset_index()
    )
    print("▶ summary.shape =", summary.shape)

    # ── H) MERGE HYPERPARAMS BACK INTO summary FOR FINAL TABLE ────────────────────
    # Merge hyperparameters from best_configs into the summary
    hyperparam_summary = best_configs[["dataset", "model"] + hyperparam_cols].copy()
    final_table = summary.merge(
        hyperparam_summary,
        on=["dataset", "model"],
        how="left"
    )
    final_cols = ["dataset", "model"] + hyperparam_cols + ["mean_test_mae", "std_test_mae"]
    final_table = final_table[final_cols]
    print("▶ final_table.shape =", final_table.shape)
    print(final_table.to_markdown(index=False, floatfmt=".4f"))
    
    # ── H) PIVOT INTO “models × datasets” ────────────────────────────────────────

    # 1) First, create a new column that formats mean±std as a single string:
    summary["mean_std"] = summary.apply(
        lambda r: f"{r['mean_test_mae']:.4f} ± {r['std_test_mae']:.4f}", axis=1
    )

    # 2) Now pivot:
    # Rank the top-k within each (dataset, model) by mean_test_mae (best first)
    summary["rank"] = (
        summary.sort_values(["dataset","model","mean_test_mae"])
            .groupby(["dataset","model"])
            .cumcount() + 1
    )

    # Pretty string
    summary["mean_std"] = summary.apply(
        lambda r: f"{r['mean_test_mae']:.4f} ± {r['std_test_mae']:.4f}", axis=1
    )

    # Pivot with two columns levels: model and rank
    pivot_table = summary.pivot(index="dataset", columns=["model","rank"], values="mean_std")

    # Optional: order the ranks as 1..k
    pivot_table = pivot_table.sort_index(axis=1, level=[0,1])

    print(pivot_table.reset_index().to_markdown(index=False))
    # pivot_table = summary.pivot(
    #     index="dataset",
    #     columns="model",
    #     values="mean_std"
    # )

    # # 3) (Optional) If you want the columns in a specific order, you can reindex:
    # #    e.g. all_datasets = ["CIFAR10", "ImageNet", ...]
    # # pivot_table = pivot_table.reindex(columns=all_datasets)

    # # 4) Reset the index (so “model” becomes a column instead of the index), if you prefer:
    # pivot_table = pivot_table.reset_index()

    # # 5) Finally, print as Markdown:
    # print("\n=== Test‐mae (mean ± std) with models as rows, datasets as columns ===\n")
    # print(pivot_table.to_markdown(index=False))
    
    return df, grouped, best_configs, summary, pivot_table


def get_latex_table(table):
    """Convert a DataFrame to a LaTeX table.
    
    Parameters:
    -----------
    table : pd.DataFrame
        DataFrame to be converted to LaTeX format.
    """
    latex_str = table.to_latex(
        index=False,
        caption="Test accuracy (mean $\pm$ std) for each dataset (rows) and model (columns)",
        label="tab:hypergraph_results",
        column_format="l" + "c" * (table.shape[1] - 1),
        escape=False  # so that the “$\pm$” math syntax is not escaped
    )
    print(latex_str)