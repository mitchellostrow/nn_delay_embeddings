from sklearn.neighbors import NearestNeighbors
import numpy as np


def find_neighbors(embedding, n_neighbors, algorithm="ball_tree"):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    indices = indices[:, 1:]  # exclude present point
    return distances, indices


def compute_EkT(data, embedding, n_neighbors, T, thresh=10):
    # data should be 1 dimensional in the last axis
    if data.ndim == 3:
        print("multiple trajectories found, recursing")
        E_kT = []
        for i in range(data.shape[0]):
            E_kT.extend(
                compute_EkT(data[i], embedding[i], n_neighbors, T, thresh=thresh)
            )
        return np.array(E_kT)

    if T < thresh:
        data = data[
            : T - thresh
        ]  # remove the last thresh-T points to keep array size consistent
        embedding = embedding[: T - thresh]
    # trajs has shape (time ,dim)
    d = embedding[:-T]  # can't look at the last T
    _, indices = find_neighbors(d, n_neighbors)
    # now given the indices, we need to find the index of the point T steps after the point of a given index
    data = data[T:]
    # calculate the mean of these points for each point d
    mu_kT = np.mean(
        data[indices], axis=1
    )  # * (n_neighbors - 1) / (n_neighbors) #unbias
    # calculate the variance of the neighborsfor each of the points d
    mu_kT = mu_kT[:, np.newaxis].repeat(n_neighbors - 1, axis=1)

    E_kT = np.mean((data[indices] - mu_kT) ** 2, axis=(1, 2))  # shape (time)

    norm = (n_neighbors - 1) / n_neighbors  # rescale
    E_kT *= norm
    return E_kT


def compute_Ek(data, embedding, n_neighbors, max_T):
    E_k = []
    for T in range(1, max_T + 1):
        E_kT = compute_EkT(data, embedding, n_neighbors, T, thresh=max_T)
        E_k.append(E_kT)
        print(E_kT.shape)
    E_k = np.array(E_k)
    E_k = np.mean(E_k, axis=0)
    return E_k


def compute_eps_k(embedding, n_neighbors, thresh=10):
    if embedding.ndim == 3:
        embedding = embedding[
            :, :-thresh
        ]  # pick the data to go along with the E_k cutoff (can't look at the last T points because we need to look T steps ahead)
        embedding = embedding.reshape(-1, embedding.shape[-1])
    else:
        embedding = embedding[:-thresh]

    distances, _ = find_neighbors(embedding, n_neighbors)

    # fact = 2 / (n_neighbors * (n_neighbors - 1))
    # this is not quite correct--need to calculate the distance between all elements in the neighborhood

    eps_k = np.sum(distances[:, 1:] ** 2, axis=1)  # * fact

    norm_factor = 1 / np.mean(1 / eps_k)
    eps_k /= distances.shape[1]
    eps_k *= 2

    return eps_k, norm_factor


def compute_noise_amp_k(
    data, embedding, n_neighbors, max_T, normalize=False
):  # noise_res=0.0,
    print("computing noise amp")
    if data.device.type == "cuda":
        data = data.cpu().numpy()
    else:
        data = data.numpy()

    if embedding.device.type == "cuda":
        embedding = embedding.detach().cpu().numpy()
    else:
        embedding = embedding.detach().numpy()

    if np.iscomplex(embedding).any():
        # hidden = np.real(embedding)
        embedding = np.concatenate([np.real(embedding), np.imag(embedding)], axis=-1)
    # if noise_res >= 0:
    # embedding += np.random.uniform(-noise_res, noise_res, embedding.shape)
    E_k = compute_Ek(data, embedding, n_neighbors + 1, max_T)

    eps_k, norm_factor = compute_eps_k(embedding, n_neighbors + 1, max_T)

    sig = np.mean(E_k / eps_k)
    if normalize:
        sig *= norm_factor

    return sig, np.mean(E_k), np.mean(eps_k)
