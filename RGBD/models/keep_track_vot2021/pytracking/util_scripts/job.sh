#!/bin/bash
uname -n; echo "Job ID: $JOB_ID"; echo "GPU: $CUDA_VISIBLE_DEVICES"
cd ..
source /scratch_net/clariden/chmayer/conda/etc/profile.d/conda.sh
conda activate pytracking
#export LD_LIBRARY_PATH="/home/damartin/scratch/software/cuda10/lib64:/home/damartin/scratch/libs/libjpeg-turbo/build/install/lib:$LD_LIBRARY_PATH"
#export PATH="/home/damartin/scratch/software/cuda10/bin:$PATH"
CUDA_HOME=/scratch_net/clariden/chmayer/libs/cuda-10.0

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# shellcheck disable=SC2046
#source /home/chmayer/.bashrc
taskset -c $(util_scripts/gpu2cpu_affinity.py $CUDA_VISIBLE_DEVICES) python -u run_experiment.py $1 $2 --threads $3
