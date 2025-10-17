fold=(0 1 2 3 4)

for fold in "${fold[@]}"; do

    python -m topobench \
        dataset=graph/FTD \
        model=graph/mlp \
        dataset.loader.parameters.adj_metric=pointcloud \
        dataset.loader.parameters.adj_thresh=1.0 \
        dataset.loader.parameters.kfold=true \
        dataset.loader.parameters.num_folds=5 \
        dataset.loader.parameters.fold=$fold \
        dataset.dataloader_params.batch_size=8,32,64 \
        dataset.loader.parameters.two_pass=true \
        dataset.loader.parameters.num_nodes=3667 \
        dataset.loader.parameters.y_val=global_cog_slope \
        model.readout.graph_encoder_dim=\[256,64\],\[128,32\],\[64,16\] \
        model.readout.feature_encoder_dim=16,8,4 \
        model.readout.fc_dim=\[128,32,16\],\[32,16,8\],\[64,32,16\] \
        model.readout.fc_dropout=0.1,0.2,0.3 \
        model.readout.fc_act=relu \
        optimizer.parameters.lr=0.001 \
        dataset.split_params.data_seed=0 \
        trainer.max_epochs=1000 \
        trainer.min_epochs=200 \
        trainer.check_val_every_n_epoch=1 \
        trainer.devices=\[0\] \
        logger.wandb.project=Proteo_new_variable_stratified_filtered \
        callbacks.early_stopping.patience=200 \
        tags="[TopoProteoGridSearch]" \
        --multirun &
    
    python -m topobench \
        dataset=graph/FTD \
        model=graph/mlp \
        dataset.loader.parameters.adj_metric=pointcloud \
        dataset.loader.parameters.adj_thresh=1.0 \
        dataset.loader.parameters.kfold=true \
        dataset.loader.parameters.num_folds=5 \
        dataset.loader.parameters.fold=$fold \
        dataset.dataloader_params.batch_size=8,32,64 \
        dataset.loader.parameters.y_val=global_cog_slope \
        dataset.loader.parameters.two_pass=true \
        dataset.loader.parameters.num_nodes=3667 \
        model.readout.graph_encoder_dim=\[256,64\],\[128,32\],\[64,16\] \
        model.readout.feature_encoder_dim=16,8,4 \
        model.readout.fc_dim=\[128,32,16\],\[32,16,8\],\[64,32,16\] \
        model.readout.fc_dropout=0.1,0.2,0.3 \
        model.readout.fc_act=tanh \
        optimizer.parameters.lr=0.001 \
        dataset.split_params.data_seed=0 \
        trainer.max_epochs=1000 \
        trainer.min_epochs=200 \
        trainer.check_val_every_n_epoch=1 \
        trainer.devices=\[1\] \
        logger.wandb.project=Proteo_new_variable_stratified_filtered \
        callbacks.early_stopping.patience=200 \
        tags="[TopoProteoGridSearch]" \
        --multirun &


    python -m topobench \
        dataset=graph/FTD \
        model=graph/mlp \
        dataset.loader.parameters.adj_metric=pointcloud \
        dataset.loader.parameters.adj_thresh=1.0 \
        dataset.loader.parameters.kfold=true \
        dataset.loader.parameters.num_folds=5 \
        dataset.loader.parameters.fold=$fold \
        dataset.dataloader_params.batch_size=8,32,64 \
        dataset.loader.parameters.two_pass=true \
        dataset.loader.parameters.num_nodes=3667 \
        dataset.loader.parameters.y_val=global_cog_slope \
        model.readout.graph_encoder_dim=\[256,64\],\[128,32\],\[64,16\] \
        model.readout.feature_encoder_dim=16,8,4 \
        model.readout.fc_dim=\[128,32,16\],\[32,16,8\],\[64,32,16\] \
        model.readout.fc_dropout=0.1,0.2,0.3 \
        model.readout.fc_act=relu,tanh \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.001 \
        dataset.split_params.data_seed=0 \
        trainer.max_epochs=1000 \
        trainer.min_epochs=200 \
        trainer.check_val_every_n_epoch=1 \
        trainer.devices=\[2\] \
        logger.wandb.project=Proteo_new_variable_stratified_filtered \
        callbacks.early_stopping.patience=200 \
        tags="[TopoProteoGridSearch]" \
        --multirun &
    

    python -m topobench \
        dataset=graph/FTD \
        model=graph/mlp \
        dataset.loader.parameters.adj_metric=pointcloud \
        dataset.loader.parameters.adj_thresh=1.0 \
        dataset.loader.parameters.kfold=true \
        dataset.loader.parameters.num_folds=5 \
        dataset.loader.parameters.fold=$fold \
        dataset.dataloader_params.batch_size=8,32,64 \
        dataset.loader.parameters.two_pass=true \
        dataset.loader.parameters.num_nodes=3667 \
        dataset.loader.parameters.y_val=global_cog_slope \
        model.readout.graph_encoder_dim=\[256,64\],\[128,32\],\[64,16\] \
        model.readout.feature_encoder_dim=16,8,4 \
        model.readout.fc_dim=\[128,32,16\],\[32,16,8\],\[64,32,16\] \
        model.readout.fc_dropout=0.1,0.2,0.3 \
        model.readout.fc_act=relu,tanh \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0001 \
        dataset.split_params.data_seed=0 \
        trainer.max_epochs=1000 \
        trainer.min_epochs=200 \
        trainer.check_val_every_n_epoch=1 \
        trainer.devices=\[3\] \
        logger.wandb.project=Proteo_new_variable_stratified_filtered \
        callbacks.early_stopping.patience=200 \
        tags="[TopoProteoGridSearch]" \
        --multirun &

    python -m topobench \
        dataset=graph/FTD \
        model=graph/mlp \
        dataset.loader.parameters.adj_metric=pointcloud \
        dataset.loader.parameters.adj_thresh=1.0 \
        dataset.loader.parameters.kfold=true \
        dataset.loader.parameters.num_folds=5 \
        dataset.loader.parameters.fold=$fold \
        dataset.dataloader_params.batch_size=8,32,64 \
        dataset.loader.parameters.y_val=global_cog_slope \
        dataset.loader.parameters.two_pass=true \
        dataset.loader.parameters.num_nodes=3667 \
        model.readout.graph_encoder_dim=\[256,64\],\[128,32\],\[64,16\] \
        model.readout.feature_encoder_dim=16,8,4 \
        model.readout.fc_dim=\[128,32,16\],\[32,16,8\],\[64,32,16\] \
        model.readout.fc_dropout=0.1,0.2,0.3 \
        model.readout.fc_act=relu \
        optimizer.parameters.lr=0.01 \
        dataset.split_params.data_seed=0 \
        trainer.max_epochs=1000 \
        trainer.min_epochs=200 \
        trainer.check_val_every_n_epoch=1 \
        trainer.devices=\[4\] \
        logger.wandb.project=Proteo_new_variable_stratified_filtered \
        callbacks.early_stopping.patience=200 \
        tags="[TopoProteoGridSearch]" \
        --multirun &

    python -m topobench \
        dataset=graph/FTD \
        model=graph/mlp \
        dataset.loader.parameters.adj_metric=pointcloud \
        dataset.loader.parameters.adj_thresh=1.0 \
        dataset.loader.parameters.kfold=true \
        dataset.loader.parameters.num_folds=5 \
        dataset.loader.parameters.fold=$fold \
        dataset.loader.parameters.two_pass=true \
        dataset.loader.parameters.num_nodes=3667 \
        dataset.dataloader_params.batch_size=8,32,64 \
        dataset.loader.parameters.y_val=global_cog_slope \
        model.readout.graph_encoder_dim=\[256,64\],\[128,32\],\[64,16\] \
        model.readout.feature_encoder_dim=16,8,4 \
        model.readout.fc_dim=\[128,32,16\],\[32,16,8\],\[64,32,16\] \
        model.readout.fc_dropout=0.1,0.2,0.3 \
        model.readout.fc_act=tanh \
        optimizer.parameters.lr=0.01 \
        dataset.split_params.data_seed=0 \
        trainer.max_epochs=1000 \
        trainer.min_epochs=200 \
        trainer.check_val_every_n_epoch=1 \
        trainer.devices=\[5\] \
        logger.wandb.project=Proteo_new_variable_stratified_filtered \
        callbacks.early_stopping.patience=200 \
        tags="[TopoProteoGridSearch]" \
        --multirun &


    python -m topobench \
        dataset=graph/FTD \
        model=graph/mlp \
        dataset.loader.parameters.adj_metric=pointcloud \
        dataset.loader.parameters.adj_thresh=1.0 \
        dataset.loader.parameters.kfold=true \
        dataset.loader.parameters.num_folds=5 \
        dataset.loader.parameters.fold=$fold \
        dataset.loader.parameters.two_pass=true \
        dataset.loader.parameters.num_nodes=3667 \
        dataset.dataloader_params.batch_size=8,32,64 \
        dataset.loader.parameters.y_val=global_cog_slope \
        model.readout.graph_encoder_dim=\[256,64\],\[128,32\],\[64,16\] \
        model.readout.feature_encoder_dim=16,8,4 \
        model.readout.fc_dim=\[128,32,16\],\[32,16,8\],\[64,32,16\] \
        model.readout.fc_dropout=0.1,0.2,0.3 \
        model.readout.fc_act=relu,tanh \
        optimizer.parameters.lr=0.01 \
        optimizer.parameters.weight_decay=0.001 \
        dataset.split_params.data_seed=0 \
        trainer.max_epochs=1000 \
        trainer.min_epochs=200 \
        trainer.check_val_every_n_epoch=1 \
        trainer.devices=\[6\] \
        logger.wandb.project=Proteo_new_variable_stratified_filtered \
        callbacks.early_stopping.patience=200 \
        tags="[TopoProteoGridSearch]" \
        --multirun &
    
    python -m topobench \
        dataset=graph/FTD \
        model=graph/mlp \
        dataset.loader.parameters.adj_metric=pointcloud \
        dataset.loader.parameters.adj_thresh=1.0 \
        dataset.loader.parameters.kfold=true \
        dataset.loader.parameters.num_folds=5 \
        dataset.loader.parameters.fold=$fold \
        dataset.loader.parameters.two_pass=true \
        dataset.loader.parameters.num_nodes=3667 \
        dataset.dataloader_params.batch_size=8,32,64 \
        dataset.loader.parameters.y_val=global_cog_slope \
        model.readout.graph_encoder_dim=\[256,64\],\[128,32\],\[64,16\] \
        model.readout.feature_encoder_dim=16,8,4 \
        model.readout.fc_dim=\[128,32,16\],\[32,16,8\],\[64,32,16\] \
        model.readout.fc_dropout=0.1,0.2,0.3 \
        model.readout.fc_act=relu,tanh \
        optimizer.parameters.lr=0.01 \
        optimizer.parameters.weight_decay=0.0001 \
        dataset.split_params.data_seed=0 \
        trainer.max_epochs=1000 \
        trainer.min_epochs=200 \
        trainer.check_val_every_n_epoch=1 \
        trainer.devices=\[6\] \
        logger.wandb.project=Proteo_new_variable_stratified_filtered \
        callbacks.early_stopping.patience=200 \
        tags="[TopoProteoGridSearch]" \
        --multirun &
done