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
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

# -- run all of the above functions and plot

def eval_embedding(
    attractor, model, full_data, y, y_pred, hiddens, cfg, verbose=True
):
    eval_cfg = cfg.eval
    dim_observed = cfg.attractor.dim_observed
    y = y.cpu().detach().numpy()

    y_pred = y_pred.cpu().detach().numpy()
    hiddens = hiddens.cpu().detach().numpy()
    if np.iscomplex(hiddens).any():
        hiddens = np.concatenate([hiddens.real, hiddens.imag], axis=-1)
    # evaluate all of the above metrics using the arguments in this funciton
    # and log them on wandb
    # compute dynamic quantities

    if 'compute_dynamic_quantities' in cfg.eval.metrics:
        attractor_stats, model_stats = compute_dynamic_quantities(
            model,
            attractor,
            eval_cfg.dyn_quants.traj_length,
            eval_cfg.dyn_quants.ntrajs,
            cfg.attractor.resample,
        )
        # # attarctor_stats and model_stats are both dictionaries, so log them in wandb
        wandb.log(attractor_stats)
        wandb.log(model_stats)
    
    if 'compute_all_pred_stats' in cfg.eval.metrics:
        if verbose:
            print("computing prediction stats")
        # compute all prediction stats
        model_dim = hiddens.shape[-1]
        pred_stats = compute_all_pred_stats(y, y_pred, model_dim)
        wandb.log(pred_stats)

    if 'neighbors_comparison' in cfg.eval.metrics:
        # neighbors comparison
        if verbose:
            print("computing neighbor comparisons")

        overlap_neighb, corr_neighb = neighbors_comparison(
            full_data, hiddens, eval_cfg.neighbors_comparison.n_neighbors
        )
        wandb.log(dict(neighbors_overlap=overlap_neighb, neighbors_correlation=corr_neighb))

    if 'gp_diff_asym' in cfg.eval.metrics:
        # GP distance
        if verbose:
            print("computing grassberg-proccacia similarity")
        gp_distance = gp_diff_asym(full_data, hiddens)
        wandb.log(dict(gp_distance=gp_distance))

    if 'predict_hidden_dims_lm' in cfg.eval.metrics:
        # predict hidden dimensions
        if verbose:
            print("predicting hidden dimensions of attractor from embedding")

        train_score, test_score, classifier = predict_hidden_dims(
            full_data,
            hiddens,
            dim_observed,
            model=ElasticNet,
            **eval_cfg.predict_hiddens.linear_model_kwargs,
        )
        wandb.log(
            dict(
                lm_predict_attractor_train_score=train_score,
                lm_predict_attractor_test_score=test_score,
            )
        )

    if 'predict_hidden_dims_mlp' in cfg.eval.metrics:
        # # MLP
        train_score, test_score, classifier = predict_hidden_dims(
            full_data,
            hiddens,
            dim_observed,
            model=MLPRegressor,
            **eval_cfg.predict_hiddens.mlp_kwargs,
        )
        wandb.log(
            dict(
                mlp_predict_attractor_train_score=train_score,
                mlp_predict_attractor_test_score=test_score,
            )
        )

    # plot the attractor and model trajectories in top n dimensions -- with 2 plots that plot dimension i against dimension j
    # for i,j in the top n dimensions
    if verbose:
        print("plotting low-d trajectories")

    # run_plot_pca(full_data, "attractor")
    run_plot_pca(hiddens, "model embedding")

    if 'dsa' in cfg.eval.metrics:
        if verbose:
            print("computing DSA")
        # DSA
        dsa_cfg = eval_cfg.dsa

        if dsa_cfg.pca_dim is not None:
            def reduce(xx,n_components):
                pca = PCA(n_components=n_components)
                d = xx.reshape(-1, xx.shape[-1])
                red = pca.fit_transform(d)
                xx = red.reshape(xx.shape[0], xx.shape[1], n_components)
                return xx
            full_data = reduce(full_data, dsa_cfg.pca_dim)
            hiddens = reduce(hiddens,dsa_cfg.pca_dim)
        del dsa_cfg.pca_dim #not a keyword in dsa
        dsa = DSA(full_data, hiddens, **dict(dsa_cfg))
        score = dsa.fit_score()  # only 1 comparison so look at that
        wandb.log(dict(dsa=score))

def eval_nstep(model,data,cfg):
    #runs the model for n steps and compares the output to the true data
    #both predictively and statistically
    dim_observed = cfg.attractor.dim_observed
    data = data[:, :, dim_observed : dim_observed + 1]
    n = cfg.eval.nsteps
    device = next(model.parameters()).device
    x = data[:, :-n]
    for i in range(1,n+1):
        x = x.to(device)
        y_pred, hiddens = model(x) 
        tn = i-n if i - n < 0 else None
        y_i = data[:,i:tn]
        x = y_pred.detach()

        if i > 1:   
            model_dim = hiddens.shape[-1]
            pred_stats = compute_all_pred_stats(y_i, y_pred.detach().cpu(), model_dim)
            for k in pred_stats:
                wandb.log({f"{i}_step_{k}": pred_stats[k]})
            #TODO: compute other stats
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
                ax[i, j].plot(red[k, :, i], red[k, :, j])
            ax[i, j].set_xlabel(f"dim {i+1}, EV {pca.explained_variance_ratio_[i]:.2f}")
            ax[i, j].set_ylabel(f"dim {j+1}, EV {pca.explained_variance_ratio_[j]:.2f}")

    fig.suptitle(label)
    plt.tight_layout()
    plt.savefig(f"{label}.pdf")
    # save in wandb
    wandb.log({label: plt})
