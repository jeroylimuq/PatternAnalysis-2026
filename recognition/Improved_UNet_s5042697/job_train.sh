#!/bin/bash
# Slurm job for training on the Rangpur cluster (COMP3710 partition).
#
# Submit from this folder. Settings come from environment variables:
#   sbatch job_train.sh                                         # full run, Improved U-Net
#   MODEL=unet sbatch job_train.sh                              # full run, baseline
#   MODEL=improved EPOCHS=2 LIMIT=500 sbatch --time=00:20:00 job_train.sh   # quick test

#SBATCH --job-name=hipmri-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=08:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

set -euo pipefail

MODEL=${MODEL:-improved}
EPOCHS=${EPOCHS:-30}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
DATA_ROOT=/home/groups/comp3710/HipMRI_Study_open/keras_slices_data

source "$HOME/miniconda3/bin/activate"
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "model=$MODEL epochs=$EPOCHS ${LIMIT_ARG}"
nvidia-smi

python train.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --data-root "$DATA_ROOT" \
    --num-workers 2 \
    $LIMIT_ARG

echo "Finished $(date)"
