DATASETS=('MUTAG' 'PROTEINS')
project_name="LIFTING_TEST"


for dataset in ${DATASETS[*]}
do
    python topobenchmark/run.py\
    model=simplicial/scn\
    dataset=graph/$dataset\
    trainer.max_epochs=2\
    trainer.min_epochs=1\
    trainer.check_val_every_n_epoch=1\
    transforms=g2s_khop
done