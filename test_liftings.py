"""Test liftings with python to be able to use the debugger more easily."""

import subprocess

# Configurations
project_name = "LIFTING_TEST"
datasets = ["geometric_shapes"]  # Add more datasets if needed
transforms_experiments = ["exp_combinatorial/h2c_universal_strict"]

# Run the commands in Python
for transform in transforms_experiments:
    for dataset in datasets:
        command = [
            "python",
            "topobench/run.py",
            f"model=cell/topotune",
            f"dataset=hypergraph/cocitation_cora",
            f"trainer.max_epochs=2",
            f"trainer.min_epochs=1",
            f"trainer.check_val_every_n_epoch=1",
            f"transforms={transform}",
            f"logger.wandb.project={project_name}",
        ]

        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
