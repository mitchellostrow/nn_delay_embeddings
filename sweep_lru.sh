#!/bin/bash
#SBATCH --job-name=sweep_lru
#SBATCH -N 1         
#SBATCH -n 2 # n CPU (hyperthreaded) cores
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --partition=normal
#SBATCH --gres=gpu:0

unset XDG_RUNTIME_DIR
source activate nn_delays
python train.py -m model=lru data.time=500 model.kwargs.d_model=10,25,50,100 train.schedule=True,False model.kwargs.mlp_hidden=50,100 model.kwargs.siso=False attractor.observed_noise=0.0,0.05
