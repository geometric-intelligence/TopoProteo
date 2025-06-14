#!/bin/bash -l
pip install --upgrade pip
pip install -e '.[all]'

# Note that not all combinations of torch and CUDA are available
# See https://github.com/pyg-team/pyg-lib to check the configuration that works for you
TORCH="2.3.0"   # available options: 1.12.0, 1.13.0, 2.0.0, 2.1.0, 2.2.0, or 2.3.0
CUDA="cu118"    # if available, select the CUDA versicn suitable for your system
                # available options: cpu, cu102, cu113, cu116, cu117, cu118, or cu121
pip install --force-reinstall torch==${TORCH} --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install --force-reinstall pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

#pytest

pre-commit install
