import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# basic training loop to train a model


def train(model, train_set, val_set, epochs, lr, optimizer, loss_fn, device, nsteps=1):
    """
    Trains a model on a dataset
    """
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)
    train_loss = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for i, data in enumerate(train_set):

            x = data[:, :-nsteps]
            y = data[:, nsteps:]

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            for _ in range(nsteps):
                y_pred, _ = model(x)
                x = y_pred
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss / len(train_set))
        print(f"Epoch {epoch} Training Loss: {total_loss/len(train_set)}")
        model.eval()
        val_loss = 0
        for i, data in enumerate(val_set):
            x = data[:, :-1]
            y = data[:, 1:]
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
        val_losses.append(val_loss / len(val_set))

        print(f"Epoch {epoch} Validation Loss: {val_loss/len(val_set)}")
    return model, train_loss, val_losses


def make_dataset(system, length, nsamples):
    # convert these into dataloaders

    def gen_data():
        model = system()
        model.ic = model.ic[None, :] * np.random.random(nsamples)[:, None]
        data = model.make_trajectory(length, resample=True)
        data_used = torch.tensor(data).float()[:, :, :1]
        print(data.shape, data_used.shape)
        loader = DataLoader(data_used, batch_size=1, shuffle=True)
        return loader, data

    train_set, train_data = gen_data()
    val_set, val_data = gen_data()

    return train_set, val_set, train_data, val_data



def embed_signal_torch(data, n_delays, delay_interval=1):
    """
    Create a delay embedding from the provided tensor data.

    Parameters
    ----------
    data : torch.tensor
        The data from which to create the delay embedding. Must be either: (1) a
        2-dimensional array/tensor of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    n_delays : int
        Parameter that controls the size of the delay embedding. Explicitly,
        the number of delays to include.

    delay_interval : int
        The number of time steps between each delay in the delay embedding. Defaults
        to 1 time step.
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    device = data.device

    if data.shape[int(data.ndim == 3)] - (n_delays - 1) * delay_interval < 1:
        raise ValueError(
            "The number of delays is too large for the number of time points in the data!"
        )

    # initialize the embedding
    if data.ndim == 3:
        embedding = torch.zeros(
            (
                data.shape[0],
                data.shape[1] - (n_delays - 1) * delay_interval,
                data.shape[2] * n_delays,
            )
        ).to(device)
    else:
        embedding = torch.zeros(
            (data.shape[0] - (n_delays - 1) * delay_interval, data.shape[1] * n_delays)
        ).to(device)

    for d in range(n_delays):
        index = (n_delays - 1 - d) * delay_interval
        ddelay = d * delay_interval

        if data.ndim == 3:
            ddata = d * data.shape[2]
            embedding[:, :, ddata : ddata + data.shape[2]] = data[
                :, index : data.shape[1] - ddelay
            ]
        else:
            ddata = d * data.shape[1]
            embedding[:, ddata : ddata + data.shape[1]] = data[
                index : data.shape[0] - ddelay
            ]

    return embedding