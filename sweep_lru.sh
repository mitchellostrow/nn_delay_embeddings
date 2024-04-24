#!/bin/bash
#SBATCH --job-name=sweep_lru
#SBATCH -N 1         
#SBATCH -n 2 # n CPU (hyperthreaded) cores
#SBATCH --time=2-0:00:00
#SBATCH --mem=2GB
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

unset XDG_RUNTIME_DIR
conda activate nn_delays
python train.py -m model=lru model.kwargs.d_model=10,50,100 model.kwargs.d_state=10,50,100,200 model.kwargs.expansion=1,4