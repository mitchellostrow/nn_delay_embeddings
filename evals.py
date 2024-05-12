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
from sklearn.manifold import Isomap
import io
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
import torch
from scipy.ndimage import gaussian_filter1d

# -- run all of the above functions and plot


def eval_embedding(attractor, model, full_data, y, y_pred, hiddens, cfg, verbose=True):
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

    if "compute_dynamic_quantities" in cfg.eval.metrics:
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

    if "compute_all_pred_stats" in cfg.eval.metrics:
        if verbose:
            print("computing prediction stats")
        # compute all prediction stats
        model_dim = hiddens.shape[-1]
        pred_stats = compute_all_pred_stats(y, y_pred, model_dim)
        wandb.log(pred_stats)

    if "neighbors_comparison" in cfg.eval.metrics:
        # neighbors comparison
        if verbose:
            print("computing neighbor comparisons")

        overlap_neighb, corr_neighb = neighbors_comparison(
            full_data, hiddens, eval_cfg.neighbors_comparison.n_neighbors
        )
        wandb.log(
            dict(neighbors_overlap=overlap_neighb, neighbors_correlation=corr_neighb)
        )

    if "gp_diff_asym" in cfg.eval.metrics and not np.isnan(
        model_stats["model_estim_lyap"]
    ):
        # GP distance
        if verbose:
            print("computing grassberg-proccacia similarity")
        gp_distance = gp_diff_asym(full_data, hiddens)
        wandb.log(dict(gp_distance=gp_distance))

    if "predict_hidden_dims_lm" in cfg.eval.metrics:
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

    if "predict_hidden_dims_mlp" in cfg.eval.metrics:
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

    if "dsa" in cfg.eval.metrics:
        if verbose:
            print("computing DSA")
        # DSA
        dsa_cfg = eval_cfg.dsa

        if dsa_cfg.pca_dim is not None:

            def reduce(xx, n_components):
                pca = PCA(n_components=n_components)
                d = xx.reshape(-1, xx.shape[-1])
                red = pca.fit_transform(d)
                xx = red.reshape(xx.shape[0], xx.shape[1], n_components)
                return xx

            full_data = reduce(full_data, dsa_cfg.pca_dim)
            hiddens = reduce(hiddens, dsa_cfg.pca_dim)
        dss_cfg = {k: v for k, v in dict(dsa_cfg).items() if k != "pca_dim"}
        dsa = DSA(full_data, hiddens, **dss_cfg)
        score = dsa.fit_score()  # only 1 comparison so look at that
        wandb.log(dict(dsa=score))


def eval_nstep(model, data, cfg, epoch):
    # runs the model for n steps and compares the output to the true data
    # both predictively and statistically
    dim_observed = cfg.attractor.dim_observed
    data = data[:, :, dim_observed : dim_observed + 1]
    n = cfg.eval.nstep_eval.nsteps
    device = next(model.parameters()).device
    x = data[:, :-n]
    observed = np.linspace(1, n + 1, cfg.eval.nstep_eval.n_obs, dtype=int)
    plt.figure()
    plt.plot(data[0, :, 0], label="true", c="k")
    for i in range(1, n + 1):  # only observe a few
        x = x.to(device)
        y_pred, hiddens = model(x)
        tn = i - n if i - n < 0 else None
        y_i = data[:, i:tn]
        x = y_pred.detach()
        if i in observed:
            # plot y_i and y_pred over time, index starting from i,
            plt.plot(
                np.arange(i, i + y_pred.shape[1]),
                y_pred[0, :, 0].cpu().detach().numpy(),
                label=f"{i}-step pred",
            )

            model_dim = hiddens.shape[-1]
            pred_stats = compute_all_pred_stats(y_i, y_pred.detach().cpu(), model_dim)
            for k in pred_stats:
                wandb.log({f"{i}_step_{k}": pred_stats[k]})

            # kl-div between the output trajectories
            kldiv = klx_metric(y_pred.detach().cpu(), y_i, cfg.eval.nstep_eval.kl_bins)
            wandb.log({f"{i}_step_kl_div": kldiv})

            # correlation between fourier spectra
            spectral_corr = np.mean(
                power_spectrum_error_per_dim(
                    y_pred.detach().cpu(),
                    y_i,
                    cfg.eval.nstep_eval.spectral_smoothing,
                    cfg.eval.nstep_eval.spectral_cutoff,
                )
            )
            wandb.log({f"{i}_step_spectral_corr": spectral_corr})

    plt.xlabel("timestep")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"nstep_eval_epoch{epoch}.pdf")
    # wandb.log({"nstep_eval": plt})
    plt.close()


# flatten top 2 dimensions of full_data and hiddens, run pca
def run_plot_pca(data, label):
    pca = PCA(n_components=4)
    d = data.reshape(-1, data.shape[-1])
    red = pca.fit_transform(d)
    red = red.reshape(data.shape[0], data.shape[1], 4)

    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        for j in range(4):
            for k in range(min(30, data.shape[0])):
                ax[i, j].plot(red[k, :, i], red[k, :, j])
            ax[i, j].set_xlabel(f"dim {i+1}, EV {pca.explained_variance_ratio_[i]:.4f}")
            ax[i, j].set_ylabel(f"dim {j+1}, EV {pca.explained_variance_ratio_[j]:.4f}")

    fig.suptitle(label)
    plt.tight_layout()
    plt.savefig(f"{label}.pdf")
    # save in wandb
    # wandb.log({label: plt})
    plt.close()

    # plot 2d isomap too!
    red = Isomap(n_components=2, n_neighbors=15).fit_transform(d)
    red = red.reshape(data.shape[0], data.shape[1], 2)
    for k in range(4):
        plt.plot(red[k, :, 0], red[k, :, 1])
    plt.xlabel("Isomap 1")
    plt.ylabel("Isomap 2")
    plt.tight_layout()
    plt.savefig(f"{label}_isomap.pdf")
    # wandb.log({f"{label}_isomap": plt})
    plt.close()


# the following functions were adapted from https://github.com/DurstewitzLab/dendPLRNN/blob/main/BPTT_TF/evaluation/
# adapted to handle batch sizes too (collecting a full histogram across batches and then doing kl on that)


def kullback_leibler_divergence(p1, p2):
    """
    Calculate Kullback-Leibler divergence
    """
    if p1 is None or p2 is None:
        kl = torch.tensor([float("nan")])
    else:
        kl = (p1 * torch.log(p1 / p2)).sum()
    return kl


def calc_histogram(x, n_bins, min_, max_):
    """
    Calculate a multidimensional histogram in the range of min and max
    works by aggregating values in sparse tensor,
    then exploits the fact that sparse matrix indices may contain the same coordinate multiple times,
    the matrix entry is then the sum of all values at the coordinate
    for reference: https://discuss.pytorch.org/t/histogram-function-in-pytorch/5350/9
    Outliers are discarded!
    :param x: multidimensional data: shape (N, D) with N number of entries, D number of dims
    :param n_bins: number of bins in each dimension
    :param min_: minimum value
    :param max_: maximum value to consider for histogram
    :return: histogram
    """
    dim_x = x.shape[1]  # number of dimensions
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    coordinates = (n_bins * (x - min_) / (max_ - min_)).long()

    # discard outliers
    coord_bigger_zero = coordinates > 0
    coord_smaller_nbins = coordinates < n_bins
    inlier = coord_bigger_zero.all(1) * coord_smaller_nbins.all(1)
    coordinates = coordinates[inlier]

    size_ = tuple(n_bins for _ in range(dim_x))
    indices = torch.ones(coordinates.shape[0], device=coordinates.device)
    if "cuda" == coordinates.device.type:
        tens = torch.cuda.sparse.FloatTensor
    else:
        tens = torch.sparse.FloatTensor
    return tens(coordinates.t(), indices, size=size_).to_dense()


def get_min_max_range(x_true):
    std = x_true.std(0)
    return -2 * std, 2 * std


def normalize_to_pdf_with_laplace_smoothing(histogram, n_bins, smoothing_alpha=10e-6):
    if histogram.sum() == 0:  # if no entries in the range
        pdf = None
    else:
        dim_x = len(histogram.shape)
        pdf = (histogram + smoothing_alpha) / (
            histogram.sum() + smoothing_alpha * n_bins**dim_x
        )
    return pdf


def get_pdf_from_timeseries(x_gen, x_true, n_bins):
    """
    Calculate spatial pdf of time series x1 and x2
    :param x_gen: multivariate time series: shape (T, dim) or (B,T,dim)
    :param x_true: multivariate time series, used for choosing range of histogram
    :param n_bins: number of histogram bins
    :return: pdfs
    """

    if x_gen.ndim == 3:
        x_gen = x_gen.reshape(-1, x_gen.shape[-1])
    if x_true.ndim == 3:
        x_true = x_true.reshape(-1, x_true.shape[-1])

    min_, max_ = get_min_max_range(x_true)
    hist_gen = calc_histogram(x_gen, n_bins=n_bins, min_=min_, max_=max_)
    hist_true = calc_histogram(x_true, n_bins=n_bins, min_=min_, max_=max_)

    p_gen = normalize_to_pdf_with_laplace_smoothing(histogram=hist_gen, n_bins=n_bins)
    p_true = normalize_to_pdf_with_laplace_smoothing(histogram=hist_true, n_bins=n_bins)
    return p_gen, p_true


def klx_metric(x_gen, x_true, n_bins=30):
    # plot_kl(x_gen, x_true, n_bins)
    p_gen, p_true = get_pdf_from_timeseries(x_gen, x_true, n_bins)
    return kullback_leibler_divergence(p_true, p_gen)


# power spectrum autocorrelation


# now, test the spectral correlation
def convert_to_decibel(x):
    x = 20 * np.log10(x)
    return x[0]


def ensure_length_is_even(x):
    n = len(x)
    if n % 2 != 0:
        x = x[:-1]
        n = len(x)
    x = np.reshape(x, (1, n))
    return x


def fft_in_decibel(x, smoothing):
    """
    Originally by: Vlachas Pantelis, CSE-lab, ETH Zurich in https://github.com/pvlachas/RNN-RC-Chaos
    Calculate spectrum in decibel scale,
    scale the magnitude of FFT by window and factor of 2, because we are using half of FFT spectrum.
    :param x: input signal
    :return fft_decibel: spectrum in decibel scale
    """
    x = ensure_length_is_even(x)
    fft_real = np.fft.rfft(x)
    fft_magnitude = np.abs(fft_real) * 2 / len(x)
    fft_decibel = convert_to_decibel(fft_magnitude)

    fft_smoothed = gaussian_filter1d(fft_decibel, sigma=smoothing)
    return fft_smoothed


def get_average_spectrum(trajectories, smoothing):
    spectrum = []
    for trajectory in trajectories:
        trajectory = (
            trajectory - trajectory.mean()
        ) / trajectory.std()  # normalize individual trajectories
        fft_decibel = fft_in_decibel(trajectory, smoothing)
        spectrum.append(fft_decibel)
    spectrum = np.array(spectrum).mean(axis=0)
    return spectrum


def power_spectrum_error_per_dim(x_gen, x_true, smoothing, cutoff):

    x_true = x_true.reshape(x_gen.shape)

    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]
    dim_x = x_gen.shape[2]
    pse_corrs_per_dim = []
    for dim in range(dim_x):
        spectrum_true = get_average_spectrum(x_true[:, :, dim], smoothing)
        spectrum_gen = get_average_spectrum(x_gen[:, :, dim], smoothing)
        spectrum_true = spectrum_true[:cutoff]
        spectrum_gen = spectrum_gen[:cutoff]

        pse_corr_per_dim = np.abs(np.corrcoef(x=spectrum_gen, y=spectrum_true)[0, 1])
        pse_corrs_per_dim.append(pse_corr_per_dim)
    return pse_corrs_per_dim
