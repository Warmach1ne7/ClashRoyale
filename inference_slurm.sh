#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=50g
#SBATCH -J "tower_detection_yolo"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1

# Load CUDA modules
module load cuda12.6/toolkit
module load cuda12.6/blas
module load cuda12.6/fft
# Activate conda environment
conda activate clashroyale

# GPU debug info
gpu_debug

# Run training
/home/ostikar/.conda/envs/clashroyale/bin/python inference.py