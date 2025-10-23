#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -L jobenv=singularity
#PJM -j
#PJM -S

module load gcc-toolset/13 
module load cuda/12.2.2
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rapids-25.10

nsys profile -f true -o result -t cublas,cuda,cudnn,nvtx,openacc,osrt \
python main_shan.py
