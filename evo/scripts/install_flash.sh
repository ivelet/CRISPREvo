#!/bin/bash
#SBATCH --job-name=install_requirements
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH -A bi-dlc2
#SBATCH --export=NONE
#SBATCH --output=./logs/requirements-%j.out
#SBATCH --error=./logs/requirements-%j.err

source /home/koeksalr/miniforge3/bin/activate crispr-evo
pip --version
pip show torch
conda list

# pip install flash_attn==2.7.4.post1 --no-build-isolation

# Download and install prebuilt flash_attn wheel if the above does not work
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

