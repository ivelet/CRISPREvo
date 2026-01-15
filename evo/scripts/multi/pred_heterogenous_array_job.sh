#!/bin/bash
#SBATCH --job-name=install_requirements
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH -A bi-dlc2
#SBATCH --export=NONE
#SBATCH --output=./logs/requirements-%j.out
#SBATCH --error=./logs/requirements-%j.err

source /home/koeksalr/miniforge3/bin/activate crispr-evo

# Run the python script and pass the argument
python3 /work/dlclarge2/koeksalr-crispr/evo/src/multi/pred_heterogenous_array.py $1
