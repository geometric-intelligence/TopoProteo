python -m topobench \
    dataset=graph/FTD \
    model=graph/mlp \
    dataset.loader.parameters.adj_metric=pointcloud \
    dataset.loader.parameters.adj_thresh=1.0 \
    dataset.loader.parameters.kfold=true \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.y_val=global_cog_slope \
    dataset.loader.parameters.fold=0 \
    dataset.split_params.data_seed=0 \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[0\] \
    logger.wandb.project=prerun \
    callbacks.early_stopping.patience=50 \
    tags="[TopoProteoGridSearch]" \
    --multirun &

python -m topobench \
    dataset=graph/FTD \
    model=graph/mlp \
    dataset.loader.parameters.adj_metric=pointcloud \
    dataset.loader.parameters.adj_thresh=1.0 \
    dataset.loader.parameters.kfold=true \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.y_val=global_cog_slope \
    dataset.loader.parameters.fold=1 \
    dataset.split_params.data_seed=0 \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[0\] \
    logger.wandb.project=prerun \
    callbacks.early_stopping.patience=50 \
    tags="[TopoProteoGridSearch]" \
    --multirun &

python -m topobench \
    dataset=graph/FTD \
    model=graph/mlp \
    dataset.loader.parameters.adj_metric=pointcloud \
    dataset.loader.parameters.adj_thresh=1.0 \
    dataset.loader.parameters.kfold=true \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.y_val=global_cog_slope \
    dataset.loader.parameters.fold=2 \
    dataset.split_params.data_seed=0 \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[0\] \
    logger.wandb.project=prerun \
    callbacks.early_stopping.patience=50 \
    tags="[TopoProteoGridSearch]" \
    --multirun &

python -m topobench \
    dataset=graph/FTD \
    model=graph/mlp \
    dataset.loader.parameters.adj_metric=pointcloud \
    dataset.loader.parameters.adj_thresh=1.0 \
    dataset.loader.parameters.kfold=true \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.y_val=global_cog_slope \
    dataset.loader.parameters.fold=3 \
    dataset.split_params.data_seed=0 \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[3\] \
    logger.wandb.project=prerun \
    callbacks.early_stopping.patience=50 \
    tags="[TopoProteoGridSearch]" \
    --multirun &

python -m topobench \
    dataset=graph/FTD \
    model=graph/mlp \
    dataset.loader.parameters.adj_metric=pointcloud \
    dataset.loader.parameters.adj_thresh=1.0 \
    dataset.loader.parameters.kfold=true \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.y_val=global_cog_slope \
    dataset.loader.parameters.fold=4 \
    dataset.split_params.data_seed=0 \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[4\] \
    logger.wandb.project=prerun \
    callbacks.early_stopping.patience=50 \
    tags="[TopoProteoGridSearch]" \
    --multirun &
