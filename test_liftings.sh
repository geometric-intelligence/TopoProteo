project_name="LIFTING_TEST"
DATASETS=('MUTAG') #'PROTEINS'

transforms_experiments=(simplicial_experiments/g2h_neighborhood_complex)

# Date 24/02
# working: g2s_khop g2s_vietoris_rips g2h_neighborhood_complex g2s_graph_induceds g2s_dnd

# NOTE 1: the code of this submission is good but g2s_line (line lifting) construct line graph and finds graph induced topology. Hence the edges becomes nodes. THis lifting do not fits into our TB framework without modification 
# NOTE 2: it seems that g2s_graph_induced is our clique lifting. 
# NOTE 3: g2s_dnd is very slow as it generates enormous amout of simplices, even for small datasets.


for transform in ${transforms_experiments[*]}
do
    for dataset in ${DATASETS[*]}
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

