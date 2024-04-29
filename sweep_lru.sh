#!/bin/bash
#SBATCH --job-name=sweep_lru
#SBATCH -N 1         
#SBATCH -n 2 # n CPU (hyperthreaded) cores
#SBATCH --time=2-0:00:00
#SBATCH --mem=2GB
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

unset XDG_RUNTIME_DIR
source activate nn_delays
python train.py -m model=lru model.kwargs.d_model=10,25 model.kwargs.d_state=200,300,400 model.kwargs.expansion=1 train.schedule=True,False
