# file with a list of functions that will be used to evaluate models' predictive
# capabilities and embedding quality, then plotting and saving in wandb
import wandb
import matplotlib.pyplot as plt
from src.metrics import (
    compute_dynamic_quantities,
    compute_all_pred_stats,
    neighbors_comparison,
    gp_diff_asym,
    predict_hidden_dims,
)
from DSA import DSA
from sklearn.decomposition import PCA
import io
import numpy as np

# -- run all of the above functions and plot?


def eval_embedding(
    attractor, model, full_data, x, y, y_pred, hiddens, cfg, verbose=True
):
    eval_cfg = cfg.eval
    dim_observed = cfg.attractor.dim_observed
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    hiddens = hiddens.cpu().detach().numpy()
    if np.iscomplex(hiddens).any():
        hiddens = np.concatenate([hiddens.real, hiddens.imag], axis=-1)
    # evaluate all of the above metrics using the arguments in this funciton
    # and log them on wandb
    # compute dynamic quantities

    attractor_stats, model_stats = compute_dynamic_quantities(
        model,
        attractor,
        eval_cfg.dyn_quants.traj_length,
        eval_cfg.dyn_quants.ntrajs,
        cfg.attractor.resample,
    )
    # attarctor_stats and model_stats are both dictionaries, so log them in wandb
    wandb.log(attractor_stats)
    wandb.log(model_stats)
    if verbose:
        print("computing prediction stats")
    # compute all prediction stats
    model_dim = hiddens.shape[-1]
    pred_stats = compute_all_pred_stats(y, y_pred, model_dim)
    wandb.log(pred_stats)

    # neighbors comparison
    if verbose:
        print("computing neighbor comparisons")

    overlap_neighb, corr_neighb = neighbors_comparison(
        full_data, hiddens, eval_cfg.neighbors_comparison.n_neighbors
    )
    wandb.log(dict(neighbors_overlap=overlap_neighb, neighbors_correlation=corr_neighb))

    # GP distance
    # if verbose:
    #     print("computing grassberg-proccacia similarity")
    # TODO: fix this and also figure out what it's for
    # gp_distance = gp_diff_asym(full_data, hiddens)
    # wandb.log(dict(gp_distance=gp_distance))

    # predict hidden dimensions
    if verbose:
        print("predicting hidden dimensions of attractor from embedding")

    train_score, test_score = predict_hidden_dims(
        full_data, hiddens, dim_observed, **eval_cfg.predict_hiddens.model_kwargs
    )
    wandb.log(
        dict(
            predict_attractor_train_score=train_score,
            predict_attractor_test_score=test_score,
        )
    )

    if verbose:
        print("computing DSA")
    # DSA
    dsa_cfg = eval_cfg.dsa
    dsa = DSA(full_data, hiddens, **dict(dsa_cfg))
    score = dsa.fit_score()  # only 1 comparison so look at that
    wandb.log(dict(dsa=score))

    # plot the attractor and model trajectories in top n dimensions -- with 2 plots that plot dimension i against dimension j
    # for i,j in the top n dimensions
    if verbose:
        print("plotting low-d trajectories")

    # run_plot_pca(full_data, "attractor")
    run_plot_pca(hiddens, "model embedding")


# flatten top 2 dimensions of full_data and hiddens, run pca
def run_plot_pca(data, label):
    pca = PCA(n_components=4)
    d = data.reshape(-1, data.shape[-1])
    red = pca.fit_transform(d)
    red = red.reshape(data.shape[0], data.shape[1], 4)

    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ax[i, j].plot(red[k, i], red[k, j])
            ax[i, j].set_xlabel(f"dim {i+1}, EV {pca.explained_variance_ratio_[i]:.2f}")
            ax[i, j].set_ylabel(f"dim {j+1}, EV {pca.explained_variance_ratio_[j]:.2f}")

    fig.suptitle(label)
    plt.tight_layout()
    plt.savefig(f"{label}.pdf")
    # save in wandb
    wandb.log({label: plt})
