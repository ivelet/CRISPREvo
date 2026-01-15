#!/bin/bash
#SBATCH --job-name=download_seqs
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH -A bi-dlc2
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --export=NONE
#SBATCH --output=./logs/download_seqs/slurm-%j.out
#SBATCH --error=./logs/download_seqs/slurm-%j.err

source /home/koeksalr/miniforge3/bin/activate crispr-evo
python3 /work/dlclarge2/koeksalr-crispr/NCBI-Download/downloader.py
