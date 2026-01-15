#!/bin/bash
#SBATCH --job-name=download_seqs
#SBATCH --partition=bldlc2_gpu-h200
#SBATCH --time=1:00:00:00
#SBATCH --ntasks=1
#SBATCH -A bi-dlc2
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --export=NONE
#SBATCH --output=./logs/download_seqs/slurm-%j.out
#SBATCH --error=./logs/download_seqs/slurm-%j.err


# Set up WandB API key
# export WANDB_API_KEY=$(cat ./wand_api_key.txt)

source /home/koeksalr/miniforge3/bin/activate crispr-evo

python3 /work/dlclarge2/koeksalr-crispr/evo/src/multi/train.py