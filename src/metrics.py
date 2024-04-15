from dysts import metrics
import numpy as np


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
    else: # true_vals.ndim == 3
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

    return ((x - y)**2).mean()

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
    
    SS_res = np.sum((true_vals - pred_vals)**2, dim=0)
    SS_tot = np.sum((true_vals - np.mean(true_vals, dim=0))**2, dim=0)

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
    AIC = (N*np.log(((x - y)**2).sum()/N) + 2*(rank*rank + 1))

    if norm:
        AIC /= N

    return AIC

def log_mse(x,y,norm=True):

    N = np.prod(x.shape)
    logmse = (N*np.log(((x - y)**2).sum()/N))
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
        "logMSE": log_mse(true_vals,pred_vals,norm=norm)
    }
