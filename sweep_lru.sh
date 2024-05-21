#!/bin/bash
#SBATCH --job-name=sweep_lru
#SBATCH -N 1         
#SBATCH -n 2 # n CPU (hyperthreaded) cores
#SBATCH --time=2-0:00:00
#SBATCH --mem=2GB
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:0

unset XDG_RUNTIME_DIR
source activate nn_delays
python train.py -m model=lru model.kwargs.d_model=10,25,50,100 model.kwargs.d_state=10,25,50,100 train.schedule=True,False model.kwargs.mlp_hidden=10,25,50,100 model.kwargs.siso=True,False attractor.observed_noise=0.0
