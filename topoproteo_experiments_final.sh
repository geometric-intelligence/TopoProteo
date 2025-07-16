


python -m topobench \
    dataset=graph/FTD \
    model=graph/gcn \
    dataset.loader.parameters.adj_metric=spearman_correlation \
    dataset.loader.parameters.adj_thresh=0.60 \
    dataset.loader.parameters.kfold=false \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.fold=0 \
    dataset.dataloader_params.batch_size=32 \
    model.feature_encoder.out_channels=4 \
    model.backbone.dropout=0.25 \
    model.backbone.act=tanh \
    model.backbone.num_layers=2 \
    model.readout.graph_encoder_dim=128 \
    model.readout.feature_encoder_dim=64 \
    model.readout.fc_dim=\[256,128,64\] \
    model.readout.fc_dropout=0.25 \
    model.readout.fc_act=tanh \
    optimizer.parameters.lr=0.001 \
    dataset.split_params.data_seed=0 \
    trainer.max_epochs=1000 \
    trainer.min_epochs=100 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[4\] \
    logger.wandb.project=ProteoFinal \
    callbacks.early_stopping.patience=50 \
    tags="[TopoProteoGridSearch]" \
    --multirun &


python -m topobench \
    dataset=graph/FTD \
    model=graph/gcn \
    dataset.loader.parameters.adj_metric=wgcna \
    dataset.loader.parameters.adj_thresh=0.2 \
    dataset.loader.parameters.kfold=false \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.fold=0 \
    dataset.dataloader_params.batch_size=8 \
    model.feature_encoder.out_channels=4 \
    model.backbone.dropout=0.25 \
    model.backbone.act=relu \
    model.backbone.num_layers=2 \
    model.readout.graph_encoder_dim=128 \
    model.readout.feature_encoder_dim=64 \
    model.readout.fc_dim=\[256,128,64\] \
    model.readout.fc_dropout=0.25 \
    model.readout.fc_act=tanh \
    optimizer.parameters.lr=0.001 \
    dataset.split_params.data_seed=0 \
    trainer.max_epochs=1000 \
    trainer.min_epochs=100 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[5\] \
    logger.wandb.project=ProteoFinal \
    callbacks.early_stopping.patience=50 \
    tags="[TopoProteoGridSearch]" \
    --multirun &

python -m topobench \
    dataset=graph/FTD \
    model=graph/mlp \
    dataset.loader.parameters.adj_metric=spearman_correlation \
    dataset.loader.parameters.adj_thresh=0.50 \
    dataset.loader.parameters.kfold=false \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.fold=0 \
    dataset.dataloader_params.batch_size=16 \
    model.readout.graph_encoder_dim=\[512,256\] \
    model.readout.feature_encoder_dim=64 \
    model.readout.fc_dim=\[128,64,32\] \
    model.readout.fc_dropout=0.25 \
    model.readout.fc_act=tanh \
    optimizer.parameters.lr=0.001 \
    dataset.split_params.data_seed=0 \
    trainer.max_epochs=1000 \
    trainer.min_epochs=100 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[7\] \
    logger.wandb.project=ProteoFinal \
    callbacks.early_stopping.patience=50 \
    tags="[TopoProteoGridSearch]" \
    --multirun &

# python -m topobench \
#     dataset=graph/FTD \
#     model=graph/mlp \
#     dataset.loader.parameters.adj_metric=wgcna \
#     dataset.loader.parameters.adj_thresh=0.2,0.25,0.3 \
#     dataset.loader.parameters.kfold=true \
#     dataset.loader.parameters.num_folds=5 \
#     dataset.loader.parameters.fold=0,1,2,3,4 \
#     dataset.dataloader_params.batch_size=8,16,32 \
#     model.readout.graph_encoder_dim=\[512,256\],\[256,128\] \
#     model.readout.feature_encoder_dim=64\
#     model.readout.fc_dim=\[128,64,32\],\[512,512,256,128\],\[1024,1024,512,256\] \
#     model.readout.fc_dropout=0.25 \
#     model.readout.fc_act=relu,tanh \
#     optimizer.parameters.lr=0.001,0.0001 \
#     dataset.split_params.data_seed=0 \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=100 \
#     trainer.check_val_every_n_epoch=1 \
#     trainer.devices=\[7\] \
#     logger.wandb.project=ProteoFinal \
#     callbacks.early_stopping.patience=50 \
#     tags="[TopoProteoGridSearch]" \
#     --multirun &
