import numpy as np
from scipy.sparse.linalg import eigs
import scipy.io as scio

from data_handler.preprocessing.compute_distance_matrix import compute_distance_matrix


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    ----------
    Parameters
    W: np.ndarray, shape is (N, N), N is the num of vertices
    ----------
    Returns
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis = 1))
    L = D - W
    lambda_max = eigs(L, k = 1, which = 'LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    ----------
    Parameters
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    ----------
    Returns
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''
    N = L_tilde.shape[0]
    cheb_polynomials = np.array([np.identity(N), L_tilde.copy()])
    for i in range(2, K):
        cheb_polynomials = np.append(
            cheb_polynomials,
            [2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]],
            axis=0)
    return cheb_polynomials


def get_chebyshev_polynomials(ds_conf, model_conf, dataset=None, idx_fold=None, channel_selection=None, model='MSTGCN', num_folds=10):
    if model == 'MSTGCN':
        try:
            Dis_Conn = np.load(ds_conf['path_distance_matrix'].replace('idx_fold', str(idx_fold)),
                               allow_pickle=True)  # shape:[V,V]
        except FileNotFoundError:
            compute_distance_matrix(dataset, ds_conf['path_data'], ds_conf['path_preprocessed_data'],
                                    ds_conf['path_distance_matrix'], idx_fold, num_folds=num_folds)
            Dis_Conn = np.load(ds_conf['path_distance_matrix'].replace('idx_fold', str(idx_fold)),
                               allow_pickle=True)  # shape:[V,V]
        if channel_selection is not None:
            Dis_Conn = Dis_Conn[channel_selection, :][:, channel_selection]
    elif model == 'GraphSleepNet':
        if model_conf['adj_matrix'] == 'GL':
            Dis_Conn = None
        elif model_conf['adj_matrix'] == '1':
            Dis_Conn=np.ones((ds_conf['channels'], ds_conf['channels']))
        elif model_conf['adj_matrix'] == 'random':
            Dis_Conn = np.random.rand(ds_conf['channels'], ds_conf['channels'])
        elif model_conf['adj_matrix'] == 'topk' or model_conf['adj_matrix'] == 'PLV' or model_conf['adj_matrix'] == 'DD':
            Dis_Conn = scio.loadmat(ds_conf['path_cheb'])['adj']
        else:
            assert False,'Config: check ADJ'

    if Dis_Conn is not None:
        L_DC = scaled_Laplacian(Dis_Conn)  # Calculate laplacian matrix
        return cheb_polynomial(L_DC, model_conf['cheb_k'])  # K-order Chebyshev polynomial
    else:
        return None