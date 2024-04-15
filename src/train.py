#training script that takes in a model, an attractor name, and then trains the model, saving the model at given checkpoints
#saves to wandb with plots there too!
import wandb
import torch
import torch.nn as nn
from dysts.flows import * #import all of the attractors
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import hydra
import os
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from metrics import compute_all_metrics

wandb.init(project="nn_delays")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gen_data(cfg):
    model = eval(cfg.attractor.name)()
    nsamples = cfg.data.nsamples
    time = cfg.data.time
    dim_observed = cfg.attractor.dim_observed

    model.ic = model.ic[None, :] * np.random.random(nsamples)[:, None]
    data = model.make_trajectory(time,
                                resample=cfg.attractor.resample,
                                noise=cfg.attractor.driven_noise)
    
    data += np.random.randn(*data.shape) * cfg.data.noise

    data_used = torch.tensor(data).float()[:,:,dim_observed:dim_observed+1]
    print(data.shape, data_used.shape)
    loader = DataLoader(data_used, batch_size=cfg.data.batch_size, shuffle=True)
    return loader, data


def train(model,train_set,val_set,epochs,optimizer,loss_fn,device,nsteps=1):
    '''
    Trains a model on a dataset
    '''
    if isinstance(nsteps,int):
        nsteps = [nsteps]

    model.to(device)
    train_loss = []
    val_losses = []
    for n in nsteps:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for i,data in enumerate(train_set):
                
                x = data[:,:-n]
                y = data[:,n:]

                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                for _ in range(n):
                    y_pred,_ = model(x)
                    x = y_pred
                loss = loss_fn(y_pred,y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss.append(total_loss/len(train_set))

            #log on wandb instead
            wandb.log({"train_loss":total_loss/len(train_set)})

            if epoch % 10 == 0:
                model.eval()
                val_loss = 0
                for i,data in enumerate(val_set):
                    x = data[:,:-1]
                    y = data[:,1:]
                    x = x.to(device)
                    y = y.to(device)
                    y_pred,_ = model(x)
                    loss = loss_fn(y_pred,y)
                    val_loss += loss.item()
                val_losses.append(val_loss/len(val_set))

                wandb.log({"val_loss":val_loss/len(val_set)})

                print(f'Epoch {epoch} Validation Loss: {val_loss/len(val_set)}')
    return model,train_loss,val_losses

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model = eval(cfg.model.name)(**cfg.model.kwargs)
    model.train()
    wandb.watch(model,log_freq=100)

    #generate the data
    train_loader, train_dat = gen_data(cfg)
    val_loader, val_dat = gen_data(cfg)

    #train the model
    optimizer = eval(cfg.train.optimizer)
    optimizer = optimizer(model.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)

    loss = eval(cfg.train.loss)
    model,train_loss,val_loss = train(model,
                                      train_loader,
                                      val_loader,
                                      cfg.train.epochs,
                                      optimizer,
                                      loss,
                                      DEVICE,
                                      nsteps=cfg.train.nsteps)

    #save the model
    torch.save(model.state_dict(),os.getcwd() + f'/models/{cfg.model.name}.pt')

    #run metric evaluation
    model.eval()

    metrics = compute_all_metrics(model,val_dat,cfg.metrics.dyn,cfg.metrics.pred)

    #log the metrics
    wandb.log(metrics)


if __name__ == "__main__":
    main()