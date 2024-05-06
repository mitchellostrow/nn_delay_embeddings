#!/bin/bash
#SBATCH --job-name=sweep_rnn
#SBATCH -N 1         
#SBATCH -n 2 # n CPU (hyperthreaded) cores
#SBATCH --time=2-0:00:00
#SBATCH --mem=2GB
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

unset XDG_RUNTIME_DIR
source activate nn_delays

python train.py -m model=rnn model.kwargs.architecture=LSTM,GRU,VanillaRNN model.kwargs.d_model=10,25,50,100 train.schedule=True,False model.kwargs.seed=11,12,13,14,15
