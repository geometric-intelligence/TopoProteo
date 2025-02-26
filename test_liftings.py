"""Test liftings with python to be able to use the debugger more easily."""

import subprocess

# Configurations
project_name = "LIFTING_TEST"
datasets = ["geometric_shapes"]  # Add more datasets if needed
transforms_experiments = ["exp_hypergraph/p2h_voronoi"]

# Run the commands in Python
for transform in transforms_experiments:
    for dataset in datasets:
        command = [
            "python",
            "topobenchmark/run.py",
            f"model=hypergraph/unignn2",
            f"dataset=pointcloud/{dataset}",
            f"trainer.max_epochs=2",
            f"trainer.min_epochs=1",
            f"trainer.check_val_every_n_epoch=1",
            f"transforms={transform}",
            f"logger.wandb.project={project_name}",
        ]

        print(f"Running: {' '.join(command)}")
        subprocess.run(command)


# # ------------------------- Hypergraph2Simplicial -------------------------
# transforms_experiments=(exp_simplicial/h2s_heat)
# # Date 25/02
# # working:
# # Not working exp_simplicial/h2s_heat
# # NOTE 1: exp_simplicial/h2s_heat is not working. We can remove the dependency from the external library but the lifting doesn't return a simplicial complex but a torch_geometric.data.Data object.

# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobenchmark/run.py\
#             model=simplicial/scn\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#             #trainer.devices=\[$device\]
#     done
# done


# # ------------------------- Graph2Hypergraph -------------------------
# transforms_experiments=(exp_hypergraph/g2h_forman_ricci_curvature exp_hypergraph/g2h_expander_graph g2h_kernel  exp_hypergraph/g2h_khop g2h_knn exp_hypergraph/g2h_mapper exp_hypergraph/g2h_modularity_maximization)
# # Date 25/02
# # working: exp_hypergraph/g2h_forman_ricci_curvature exp_hypergraph/g2h_expander_graph g2h_kernel  exp_hypergraph/g2h_khop g2h_knn exp_hypergraph/g2h_mapper exp_hypergraph/g2h_modularity_maximization
# # Not working
# # NOTE 1:


# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobenchmark/run.py\
#             model=hypergraph/unignn2\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#             #trainer.devices=\[$device\]
#     done
# done

# # ------------------------- Graph2CELLULAR -------------------------
# transforms_experiments=(exp_cell/g2c_discrete_configuration_complex)

# # Date 24/02
# # working: exp_cell/g2h_discrete_configuration_complex
# # Not working:
# # NOTE 1: The discrete_configuration_complex has a


# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobenchmark/run.py\
#             model=cell/cwn\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#     done
# done

# ------------------------- Graph2SIMPLICIAL -------------------------
# transforms_experiments=(exp_simplicial/g2h_neighborhood_complex)
# # Date 24/02
# # working: g2s_khop g2s_vietoris_rips g2h_neighborhood_complex g2s_graph_induceds g2s_dnd

# # NOTE 1: the code of this submission is good but g2s_line (line lifting) construct line graph and finds graph induced topology. Hence the edges becomes nodes. THis lifting do not fits into our TB framework without modification
# # NOTE 2: it seems that g2s_graph_induced is our clique lifting.
# # NOTE 3: g2s_dnd is very slow as it generates enormous amout of simplices, even for small datasets.

# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobenchmark/run.py\
#             model=simplicial/scn\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#     done
# done
