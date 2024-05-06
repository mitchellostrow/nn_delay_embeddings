# training script that takes in a model, an attractor name, and then trains the model, saving the model at given checkpoints
# saves to wandb with plots there too!
import wandb
import torch
import torch.nn as nn
from dysts import flows
from dysts.flows import *
import dysts
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import hydra
import os
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim import AdamW, Adam, SGD
import sys

sys.path.append("/om2/user/ostrow/NN_delay_embeddings/nn_delay_embeddings/src")
from src.models import RNN, Mamba, S4, GPT, LRU
from evals import eval_embedding, eval_nstep
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_mod = {
    "RNN": RNN,
    "Mamba": Mamba,
    "S4": S4,
    "GPT": GPT,
    "LRU": LRU,
}  # bc eval wasn't working
init_flow = {"Lorenz": Lorenz}
init_optim = {"AdamW": AdamW}
init_loss = {"MSE": nn.MSELoss}


class CosineWarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def gen_data(cfg):
    model = init_flow[cfg.attractor.name]()  # eval(cfg.attractor.name)()
    if cfg.attractor.dt is not None and cfg.attractor.dt not in {"none", "None"}:
        model.dt = cfg.attractor.dt
    nsamples = cfg.data.nsamples
    time = cfg.data.time

    model.ic = model.ic[None, :] * 5 * np.random.random(nsamples)[:, None] #getting arbitrary dispersion
    data = model.make_trajectory(
        time+100, resample=cfg.attractor.resample, noise=cfg.attractor.driven_noise
    )
    data = data[:, 100:] #filter out the transient
    
    data += np.random.randn(*data.shape) * cfg.attractor.observed_noise
    # plot x and delay embedded x and y
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    for i in range(min(100, data.shape[0])):
        ax[0].scatter(data[i, :, 0], data[i, :, 1],c="k",alpha=0.5)
        ax[1].plot(data[i, :, cfg.attractor.dim_observed])
    plt.savefig("attractor.png")

    data = torch.tensor(data).float()

    # print(data.shape, data_used.shape)
    loader = DataLoader(data, batch_size=cfg.data.batch_size, shuffle=True)
    return model, loader, data


def train(
    cfg,
    attractor,
    model,
    train_set,
    val_set,
    epochs,
    optimizer,
    loss_fn,
    device,
    nsteps=1,
):
    """
    Trains a model on a dataset and evaluates it with the relevant metrics
    """
    if isinstance(nsteps, int):
        nsteps = [nsteps]
    if cfg.train.schedule is not None:
        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup=100, max_iters=epochs * len(nsteps)
        )

    model.to(device)
    dim_observed = cfg.attractor.dim_observed
    train_loss = []
    val_losses = []
    for j, n in enumerate(nsteps):
        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0
            for i, data in enumerate(train_set):
                data = data[:, :, dim_observed : dim_observed + 1]

                x = data[:, :-1]
                # y = data[:, n:]#ypred = data[:,1:1-n]

                x = x.to(device)
                for t in range(1, n + 1):
                    tn = t - n if t - n < 0 else None  # so we don't index from zero
                    y = data[:, t:tn].to(device)
                    optimizer.zero_grad()

                    y_pred, _ = model(x)
                    x = y_pred

                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
            train_loss.append(total_loss / len(train_set))
            lr_scheduler.step()  # Step per iteration
            print(f"Epoch {epoch} Training Loss: {total_loss/len(train_set)}")
            # log on wandb instead
            wandb.log({"train_loss": total_loss / len(train_set)})
            wandb.log({"lr": lr_scheduler.get_lr()[0]})
            if epoch % cfg.train.eval_nsteps == 0:
                # save the model
                torch.save(
                    model.state_dict(),
                    os.getcwd() + f"/{cfg.model.model_name}_{epoch*(j+1)}.pt",
                )

                model.eval()
                data = next(iter(val_set))
                obs_data = data[:, :, dim_observed : dim_observed + 1]

                x = obs_data[:, :-1]
                y = obs_data[:, 1:]
                x = x.to(device)
                y = y.to(device)
                y_pred, hiddens = model(x)

                # run the evals here
                # need to get the full state space of x
                eval_embedding(attractor, model, data[:, :-1], y, y_pred, hiddens, cfg)

                eval_nstep(model, data, cfg,epoch)

                loss = loss_fn(y_pred, y)
                val_loss = loss.item()
                val_losses.append(val_loss)

                wandb.log({"val_loss": val_loss / len(val_set)})

                print(f"Epoch {epoch} Validation Loss: {val_loss/len(val_set)}")
    return model, train_loss, val_losses


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    sys.path.append("/om2/user/ostrow/NN_delay_embeddings/nn_delay_embeddings/src")
    from src.models import RNN, Mamba, S4, GPT, LRU

    model = init_mod[cfg.model.model_name](
        **cfg.model.kwargs
    )  # eval(cfg.model.model_name)(**cfg.model.kwargs)
    model.train()
    # calcualte the number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    dict_cfg = {
        **cfg.model.kwargs,
        "model_name": cfg.model.model_name,
        "num_params": num_params,
        **cfg.attractor,
        **cfg.data,
        **cfg.train,
    }
    # create a random number ind
    ind = np.random.randint(9999)
    wandb.init(
        project="nn_delays",
        config=dict_cfg,
        name=f"{cfg.model.model_name}_{num_params // 1000}params_{ind}",
    )

    wandb.watch(model, log_freq=100)

    # generate the data
    attractor, train_loader, train_dat = gen_data(cfg)
    _, val_loader, val_dat = gen_data(cfg)

    # train the model
    optimizer = init_optim[cfg.train.optimizer]  # eval(cfg.train.optimizer)
    optimizer = optimizer(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    loss = init_loss[cfg.train.loss]()  # eval(cfg.train.loss)()
    model, train_loss, val_loss = train(
        cfg,
        attractor,
        model,
        train_loader,
        val_loader,
        cfg.train.epochs,
        optimizer,
        loss,
        DEVICE,
        nsteps=cfg.train.nsteps,
    )

    # run metric evaluation
    model.eval()

    # log the metrics
    # wandb.log(metrics)


if __name__ == "__main__":
    main()
