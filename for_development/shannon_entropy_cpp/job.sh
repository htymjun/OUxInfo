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
conda activate rapids_py310

cd /home/pj25001011/share/OUxInfo/ShannonEntropyEstimator/shannon_entropy_cpp

python main_shan.py
