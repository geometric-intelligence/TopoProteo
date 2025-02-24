project_name="LIFTING_TEST"
DATASETS=('MUTAG' 'PROTEINS')

transforms_experiments=(g2s_khop g2s_vietoris_rips)

for transform in ${transforms_experiments[*]}
for dataset in ${DATASETS[*]}
    do
        do
            python topobenchmark/run.py\
            model=simplicial/scn\
            dataset=graph/$dataset\
            trainer.max_epochs=2\
            trainer.min_epochs=1\
            trainer.check_val_every_n_epoch=1\
            transforms=$transform\
            logger.wandb.project=$project_name
        done
    done