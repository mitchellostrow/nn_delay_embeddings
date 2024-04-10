import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

#basic training loop to train a model

def train(model,train_set,val_set,epochs,lr,optimizer,loss_fn,device):
    '''
    Trains a model on a dataset
    '''
    model.to(device)
    optimizer = optimizer(model.parameters(),lr=lr)
    train_loss = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for i,data in enumerate(train_set):
            x = data[:,:-1]
            y = data[:,1:]

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred,_ = model(x)
            loss = loss_fn(y_pred,y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss/len(train_set))
        print(f'Epoch {epoch} Training Loss: {total_loss/len(train_set)}')
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

        print(f'Epoch {epoch} Validation Loss: {val_loss/len(val_set)}')
    return model,train_loss,val_losses


def make_dataset(system,length,nsamples):
    #convert these into dataloaders

    def gen_data():
        model = system()
        model.ic = model.ic[None, :] * np.random.random(nsamples)[:, None]
        data = model.make_trajectory(length, resample=True)
        data_used = torch.tensor(data).float()[:,:,:1]
        print(data.shape, data_used.shape)
        loader = DataLoader(data_used, batch_size=1, shuffle=True)
        return loader, data

    train_set, train_data = gen_data()
    val_set, val_data = gen_data()
    
    return train_set, val_set, train_data, val_data
