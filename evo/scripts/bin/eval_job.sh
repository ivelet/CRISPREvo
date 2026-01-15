#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --job-name=bin_eval
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=//work/dlclarge2/koeksalr-crispr/logs/bin/eval/slurm-%j.out
#SBATCH --error=//work/dlclarge2/koeksalr-crispr/logs/bin/eval/slurm-%j.err

ml devel/cuda/12.6

# Enable conda in the current shell
ml devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda activate evo

python3 //work/dlclarge2/koeksalr-crispr/src/bin/eval_metrics.py $1
