#!/bin/bash
# Script to train on FTD dataset and test on both FTD test set and Sporadic FTD (OOD)
#
# Usage: ./train_ftd_test_sporadic.sh
#
# This will:
# 1. Train a model on FTD dataset (train/val splits)
# 2. Test on FTD test set
# 3. Test on Sporadic FTD dataset (OOD test set) using the same trained model

python -m topobench \
    dataset=graph/FTD \
    model=graph/mlp \
    test_ood=True \
    dataset.loader.parameters.adj_metric=pointcloud \
    dataset.loader.parameters.adj_thresh=1.0 \
    dataset.loader.parameters.kfold=false \
    dataset.loader.parameters.num_folds=5 \
    dataset.loader.parameters.fold=0 \
    dataset.loader.parameters.y_val=cog_z_score \
    dataset.dataloader_params.batch_size=32 \
    dataset.loader.parameters.two_pass=true \
    dataset.loader.parameters.num_nodes=3667 \
    model.readout.graph_encoder_dim=\[128,32\] \
    model.readout.feature_encoder_dim=16 \
    model.readout.fc_dim=\[64,32,16\] \
    model.readout.fc_dropout=0.1 \
    model.readout.fc_act=tanh \
    model.readout.use_features=true \
    model.readout.which_layer="['sex', 'age']" \
    optimizer.parameters.lr=0.01 \
    optimizer.parameters.weight_decay=0.001 \
    dataset.split_params.data_seed=0 \
    +ood_dataset.loader._target_=topobench.data.loaders.graph.sporadic_ftd_dataset_loader.SporadicFTDDatasetLoader \
    +ood_dataset.loader.parameters.data_domain=graph \
    +ood_dataset.loader.parameters.data_type=proteomics \
    +ood_dataset.loader.parameters.data_name=SporadicFTD \
    +ood_dataset.loader.parameters.dataset_name=sporadic_ftd \
    +ood_dataset.loader.parameters.raw_file_name=cleanDat.Soma.CSFneat.macwide_allftdannotations_merged.csv \
    +ood_dataset.loader.parameters.error_protein_file_name=bimodal_aptamers_for_removal.xlsx \
    +ood_dataset.loader.parameters.two_pass_error_protein_file_name=cleanDat.Soma.CSF.Blood.PC.Age.Sex.Reg.2pass.filtered.5SDWinsor.csv \
    +ood_dataset.loader.parameters.y_val=cog_z_score \
    +ood_dataset.loader.parameters.modality=csf \
    +ood_dataset.loader.parameters.ftd_mutation="['GRN', 'MAPT', 'C9orf72', 'CTL']" \
    +ood_dataset.loader.parameters.sex="['M', 'F']" \
    +ood_dataset.loader.parameters.num_nodes=3667 \
    +ood_dataset.loader.parameters.adj_metric=pointcloud \
    +ood_dataset.loader.parameters.adj_thresh=1.0 \
    +ood_dataset.loader.parameters.kfold=false \
    +ood_dataset.loader.parameters.num_folds=5 \
    +ood_dataset.loader.parameters.fold=0 \
    +ood_dataset.loader.parameters.random_state=42 \
    +ood_dataset.loader.parameters.two_pass=true \
    +ood_dataset.loader.parameters.wgcna_minModuleSize=10 \
    +ood_dataset.loader.parameters.wgcna_mergeCutHeight=0.25 \
    +ood_dataset.loader.parameters.data_dir=/scratch/lcornelis/data/data_louisa/SporadicFTD \
    +ood_dataset.loader.parameters.ftd_root=/scratch/lcornelis/data/data_louisa/FTD \
    +ood_dataset.dataloader_params.batch_size=32 \
    +ood_dataset.dataloader_params.num_workers=0 \
    +ood_dataset.dataloader_params.pin_memory=False \
    trainer.max_epochs=1000 \
    trainer.min_epochs=300 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[6\] \
    trainer.deterministic=true \
    logger.wandb.project=ProteoFTD_Sporadic \
    callbacks.early_stopping.patience=300 \
    tags="[TopoProteoGridSearch]"

