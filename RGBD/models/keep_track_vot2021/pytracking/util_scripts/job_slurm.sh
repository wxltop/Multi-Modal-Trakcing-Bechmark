#!/bin/bash
uname -n; echo "Job ID: $JOB_ID"; echo "GPU: $CUDA_VISIBLE_DEVICES"
cd ..

source /scratch_net/clariden/chmayer/conda/etc/profile.d/conda.sh
conda activate pytracking

export TORCH_EXTENSIONS_DIR=/scratch_net/clariden/chmayer/tmp

python -u run_experiment.py $1 $2 --threads $3