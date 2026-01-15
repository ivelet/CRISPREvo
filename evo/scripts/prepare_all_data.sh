#!/bin/bash
#SBATCH --job-name=download_seqs
#SBATCH --partition=bldlc2_gpu-l40s
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH -A bi-dlc2
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --export=NONE
#SBATCH --output=./logs/download_seqs/slurm-%j.out
#SBATCH --error=./logs/download_seqs/slurm-%j.err

conda activate crispr-evo

python3 /work/dlclarge2/koeksalr-crispr/evo/src/prepare_data.py
python3 /work/dlclarge2/koeksalr-crispr/evo/src/bin/prepare_data.py
python3 /work/dlclarge2/koeksalr-crispr/evo/src/multi/prepare_data.py