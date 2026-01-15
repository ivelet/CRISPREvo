#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --job-name=bin_train
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu8
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --export=NONE
#SBATCH --output=//work/dlclarge2/koeksalr-crispr/logs/bin/train/slurm-%j.out
#SBATCH --error=//work/dlclarge2/koeksalr-crispr/logs/bin/train/slurm-%j.err

ml devel/cuda/12.6

# Set up WandB API key
export WANDB_API_KEY=$(cat /home/fr/fr_fr/fr_ls1369/master-thesis/wand_api_key.txt)

# Enable conda in the current shell
ml devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda activate evo

python3 //work/dlclarge2/koeksalr-crispr/src/bin/train.py