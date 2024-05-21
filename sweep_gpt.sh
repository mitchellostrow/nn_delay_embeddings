#!/bin/bash
#SBATCH --job-name=sweep_gpt
#SBATCH -N 1         
#SBATCH -n 2 # n CPU (hyperthreaded) cores
#SBATCH --time=2-0:00:00
#SBATCH --mem=2GB
#SBATCH --partition=fiete
#SBATCH --gres=gpu:0

unset XDG_RUNTIME_DIR
source activate nn_delays

python train.py -m model=gpt model.kwargs.d_model=10,25,50,100 model.kwargs.n_head=1,2,5,10 model.kwargs.use_pe=False model.kwargs.mlp_hidden=10,25,50,100 model.kwargs.use_pe=True,False train.schedule=True attractor.observed_noise=0.001,0.01,0.1
