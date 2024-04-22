from dysts.analysis import kaplan_yorke_dimension, mse_mv

try:
    from dysts.analysis import corr_integral

    no_corr = False
except ImportError:
    no_corr = True

from dysts.analysis import (
    sample_initial_conditions,
    calculate_lyapunov_exponent,
    gpdistance,
)

import numpy as np
import torch
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split


def mae(x, y):
    """
    Compute the mean absolute error between the provided arrays.

    Parameters
    ----------
    x : np.ndarray
        A multi-dimensional array.
    y : np.ndarray
        A multi-dimensional array - must be the same size as x.

    Returns
    -------
    mae_val : float
        The mean absolute error between the provided arrays.
    """

    return np.abs(x - y).mean()


def mase(true_vals, pred_vals):
    """
    Compute the mean absolute scaled error between the provided data. Explicitly, this
    is the mean absolute error on the predictions scaled by the mean absolut error achieved
    by taking the naive persistence baseline prediction, which is simply the value at the
    previous time step.

    true_vals : np.ndarray
        The ground truth time series. Must be either: (1) a
        2-dimensional array of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    pred_vals : np.ndarray
        The predicted time series. Must be of the same shape as true_vals.

    Returns
    -------
    mase_val : float
        The mean absolute scaled error between the provided arrays.
    """

    if true_vals.ndim == 2:
        persistence_baseline = mae(true_vals[:-1], true_vals[1:])
    else:  # true_vals.ndim == 3
        persistence_baseline = mae(true_vals[:, :-1], true_vals[:, 1:])

    return mae(true_vals, pred_vals) / persistence_baseline


def mse(x, y):
    """
    Compute the mean squared error between the provided arrays.

    Parameters
    ----------
    x : np.ndarray
        A multi-dimensional array.
    y : np.ndarray
        A multi-dimensional array - must be the same size as x.

    Returns
    -------
    mse_val : float
        The mean squared error between the provided arrays.
    """

    return ((x - y) ** 2).mean()


def r2(true_vals, pred_vals):
    """
    Compute the R-squared value between two sets of data. For arrays with multiple observed dimensions,
    the R-squared is computed separately for each dimension, and then averaged.

    true_vals : np.ndarray
        The ground truth time series. Must be either: (1) a
        2-dimensional array of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    pred_vals : np.ndarray
        The predicted time series. Must be of the same shape as true_vals.

    Returns
    -------
    r2_val : float
        The mean R-squared value for the provided sets of data.
    """

    if true_vals.ndim == 3:
        true_vals = true_vals.reshape(-1, true_vals.shape[-1])
        pred_vals = pred_vals.reshape(-1, pred_vals.shape[-1])

    SS_res = np.sum((true_vals - pred_vals) ** 2, dim=0)
    SS_tot = np.sum((true_vals - np.mean(true_vals, dim=0)) ** 2, dim=0)

    r2_vals = 1 - SS_res / SS_tot
    return np.mean(r2_vals)


def correl(x, y):
    """
    Compute the correlation between two sets of data. For arrays with multiple observed dimensions,
    the correlation is computed separately for each dimension, and then averaged.

    x : np.ndarray
        A multi-dimensional array. Must be either: (1) a
        2-dimensional array of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    y : np.ndarray
        A multi-dimensional array. Must be of the same shape as x.

    Returns
    -------
    correl_val : float
        The mean correlation value for the provided sets of data.
    """

    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])

    correls = np.zeros(x.shape[-1])
    for dim in range(x.shape[-1]):
        correls[dim] = np.corrcoef(np.vstack((x[:, dim], y[:, dim])))[0, 1]

    return correls.mean()


def aic(x, y, rank, norm=True):
    """
    Compute the Akaike information criterion (AIC) for the provided arrays. AIC attempts to
    balance a models prediction quality with the number of parameters used in the model.

    x : np.ndarray
        A multi-dimensional array.

    y : np.ndarray
        A multi-dimensional array. Must be of the same shape as x.

    rank : int
        The rank of the HAVOK model used for prediction.

    norm : bool
        If True, normalize the AIC by the number of data points in the arrays. Defaults
        to True.

    Returns
    -------
    aic_val : float
        The AIC value for the provided arrays.
    """

    N = np.prod(x.shape)
    AIC = N * np.log(((x - y) ** 2).sum() / N) + 2 * (rank * rank + 1)

    if norm:
        AIC /= N

    return AIC


def log_mse(x, y, norm=True):

    N = np.prod(x.shape)
    logmse = N * np.log(((x - y) ** 2).sum() / N)
    if norm:
        logmse /= N
    return logmse


def compute_all_pred_stats(true_vals, pred_vals, rank, norm=True):
    """
    Compute all statistics and put them in a dictionary.

    true_vals : np.ndarray
        The ground truth time series. Must be either: (1) a
        2-dimensional array of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    pred_vals : np.ndarray
        The predicted time series. Must be of the same shape as true_vals.

    rank : int
        The rank of the HAVOK model used for prediction.

    norm : bool
        If True, normalize the AIC by the number of data points in the arrays. Defaults
        to True.

    Returns
    -------
    stat_dict : dict
        All the computed statistics collected into a dictionary.
    """
    return {
        "MAE": mae(true_vals, pred_vals),
        "MASE": mase(true_vals, pred_vals),
        "MSE": mse(true_vals, pred_vals),
        "R2": r2(true_vals, pred_vals),
        "Correl": correl(true_vals, pred_vals),
        "AIC": aic(true_vals, pred_vals, rank, norm=norm),
        "logMSE": log_mse(true_vals, pred_vals, norm=norm),
    }


def calc_lyap(traj1, traj2, eps_max, tvals):
    separation = np.linalg.norm(traj1 - traj2, axis=1) / np.linalg.norm(traj1, axis=1)
    cutoff_index = np.where(separation < eps_max)[0][-1]
    traj1 = traj1[:cutoff_index]
    traj2 = traj2[:cutoff_index]
    lyap = calculate_lyapunov_exponent(traj1, traj2, dt=np.median(np.diff(tvals)))
    return lyap, cutoff_index


def get_flattened_hidden(model, inp):
    h1 = model(inp)[1].detach().numpy()[0]
    # first 1 is to read out hidden state only, second 0 is to read out the first trajectory in the batch
    if h1.ndim == 3:
        # flatten the last 2 dims
        h1 = h1.reshape(h1.shape[0], -1)
    return h1


def compute_LE_model(
    model,
    eq,
    obs_fxn=lambda x: x[:, 0:1],
    rtol=1e-3,
    atol=1e-10,
    n_samples=1000,
    traj_length=5000,
):
    # model is the neural network
    # eq is the attractor
    # obs_fxn is the function that extracts the observed data to be input into the model

    all_ic = sample_initial_conditions(
        eq,
        n_samples,
        traj_length=max(traj_length, n_samples),
        pts_per_period=15,
    )
    eps_attractor = 1e-3
    eps_model = 1e-2
    eps_max = rtol
    all_lyap_eq = []
    all_cutoffs_eq = []
    # same thing but for the NN
    all_lyap_model = []
    all_cutoffs_model = []
    traj1_tot = []
    traj2_tot = []
    for compute in ["attractor", "model"]:
        for ind, ic in enumerate(all_ic):
            np.random.seed(ind)
            eq.random_state = ind
            eq.ic = ic
            tvals, traj1 = eq.make_trajectory(
                traj_length,
                resample=True,
                return_times=True,
            )

            traj1_tot.append(deepcopy(traj1))

            if compute == "attractor":
                eq.ic = ic
                eq.ic += eps_attractor * np.random.random(eq.ic.shape)
                # *= (1 + eps_attractor * (np.random.random(eq.ic.shape) - 0.5))
                tvals, traj2 = eq.make_trajectory(
                    traj_length,
                    resample=True,
                    return_times=True,
                )

                traj2_tot.append(deepcopy(traj2))

                lyap, cutoff_index = calc_lyap(traj1, traj2, eps_max, tvals)
                all_cutoffs_eq.append(cutoff_index)
                all_lyap_eq.append(lyap)

            else:
                eq.ic = ic
                # perturb the initial conditions by a larger amount for the model
                eq.ic += eps_model * np.random.random(eq.ic.shape)

                # eq.ic *= (1 + eps_model * (np.random.random(eq.ic.shape) - 0.5))
                tvals, traj2 = eq.make_trajectory(
                    traj_length,
                    resample=True,
                    return_times=True,
                )
                # next, pass these through the model
                traj1_x = obs_fxn(traj1)
                traj2_x = obs_fxn(traj2)
                traj1_x = torch.tensor(traj1_x).float().reshape(1, -1, 1)
                traj2_x = torch.tensor(traj2_x).float().reshape(1, -1, 1)

                h1 = get_flattened_hidden(model, traj1_x)
                h2 = get_flattened_hidden(model, traj2_x)
                lyap, cutoff_index = calc_lyap(h1, h2, eps_max * 1e3, tvals)
                all_lyap_model.append(lyap)
                all_cutoffs_model.append(cutoff_index)

    all_lyap_eq = np.array(all_lyap_eq)
    all_cutoffs_eq = np.array(all_cutoffs_eq)
    all_lyap_model = np.array(all_lyap_model)
    all_cutoffs_model = np.array(all_cutoffs_model)
    print("lyap eq", all_lyap_eq)
    print("lyap model", all_lyap_model)
    return (
        all_lyap_eq,
        all_cutoffs_eq,
        all_lyap_model,
        all_cutoffs_model,
        traj1_tot,
        traj2_tot,
    )


def compute_dynamic_quantities(model, attractor, traj_length, ntrajs, use_mve=False):
    # basically what we're going to do is sampple a bunch of trajectories from the attractor
    # compute dynamical quantities on them, then pass the trajectories through the model
    # extract the hidden states on it, and then compute the same dynamical quantities on the hidden states

    print("getting lyapunov exponents")

    attractor_lyap, _, model_lyap, _, traj1_tot, traj2_tot = compute_LE_model(
        model, attractor, traj_length=traj_length, n_samples=ntrajs
    )

    print("calculating KY dim")

    # calculate the kaplan-yorke dim of attractor and model lyaps
    attractor_ky = kaplan_yorke_dimension(attractor_lyap)
    # filter nans from model_lyap, if there's nothing left skip ky dim
    model_lyap = model_lyap[~np.isnan(model_lyap)]
    if len(model_lyap) == 0:
        model_ky = "nan"
    else:
        model_ky = kaplan_yorke_dimension(model_lyap)

    print("calculating correlation integral")
    # for correlation integral, we want to generate 1 really long trajectory
    # and then pass it through the model
    # then we'll calculate the correlation integral on the hidden states
    # and the observed data
    tvals, traj1 = attractor.make_trajectory(
        traj_length * 10,
        resample=False,
        return_times=True,
    )

    traj1_x = torch.tensor(traj1).float().reshape(1, -1, 1)
    # h1 = model(traj1_x)[1].detach().numpy()[0]
    h1 = get_flattened_hidden(model, traj1_x)

    # if this is a complex float then stack dimensions
    if h1.dtype in [np.complex64, np.complex128]:
        h1 = np.hstack([h1.real, h1.imag])

    if not no_corr:
        model_corr_int = corr_integral(h1)
        attractor_corr_int = corr_integral(traj1)
    else:
        model_corr_int = -1
        attractor_corr_int = -1

    if use_mve:
        print("calculating multiscale entropy")
        attractor_multiscale_entropy = mse_mv(traj1)
        model_multiscale_entropy = mse_mv(h1)

    # put each set of stats into a separate dictionary
    attractor_stats = {
        "lyap": attractor_lyap,
        "ky": attractor_ky,
        "corr_int": attractor_corr_int,
        "multiscale_entropy": attractor_multiscale_entropy,
    }
    model_stats = {
        "lyap": model_lyap,
        "ky": model_ky,
        "corr_int": model_corr_int,
        "multiscale_entropy": model_multiscale_entropy,
    }

    return attractor_stats, model_stats


def neighbors_comparison(true, embedded, n_neighbors=5):
    # nearest neighbors in the original space
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors).fit(true)
    distances_orig, indices_orig = nn_orig.kneighbors(true)

    # nearest neighbors in the embedded space
    nn_embed = NearestNeighbors(n_neighbors=n_neighbors).fit(embedded)
    distances_embed, indices_embed = nn_embed.kneighbors(embedded)

    # compare neighborhoods
    jaccard_indices = [
        len(set(indices_orig[i]).intersection(indices_embed[i])) / n_neighbors
        for i in range(len(true))
    ]

    # Compare distance correlations
    correlation_coefficients = [
        np.corrcoef(distances_orig[i], distances_embed[i])[0, 1]
        for i in range(len(true))
    ]

    return np.mean(jaccard_indices), np.mean(correlation_coefficients)


def gp_diff_asym(true, embedded, standardize=True):
    # generalization of dysts gpdistance for comparison of trajecotries
    # but generalized to different dimensionalities and multiple batches (avged over)
    # different dimensionalities -> register (map lower dim to higher dim?) #TODO: check this
    b1, t1, d1 = true.shape
    b2, t2, d2 = embedded.shape
    assert b1 == b2
    assert t1 == t2

    # pad the smaller one with zeros
    if d1 > d2:
        return gp_diff_asym(embedded, true, standardize)  # gpdist is symmetric anyway
    # elif d1 == d2:
    #     register = False
    # else:
    register = True  # maybe always register for now? this is basically just to align

    gpdists = np.zeros(b1)
    for i in range(b1):
        gpdists[i] = gpdistance(true, embedded, standardize, register)

    return gpdists


def predict_hidden_dims(
    true, embedded, dim_observed, model=ElasticNetCV, **model_kwargs
):
    # given the true full state and the embedding from the net, try and predict
    # the unobserved states of the model
    d_true = true.shape[-1]
    hidden_true = [i for i in range(d_true) if i != dim_observed]

    hidden_true = true[..., hidden_true]

    model = model(**model_kwargs)

    if hidden_true.ndim == 3 and embedded.ndim == 3:
        hidden_true = hidden_true.reshape(-1, hidden_true.shape[-1])
        embedded = embedded.reshape(-1, embedded.shape[-1])
    # split embedded into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        embedded, hidden_true, test_size=0.2
    )

    model.fit(X_train, y_train)

    return model.score(X_train, y_train), model.score(X_test, y_test)
