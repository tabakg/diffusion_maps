import numpy as np
from numpy import linalg as la

def run_diffusion_map_dense(distance_matrix,eps = 0.2, alpha = 1., eig_lower_bound = None, eig_upper_bound = None):
    '''
    Computes the eigenvealues and eigenvectors for diffusion maps
    given a dense input.

    Args:
        distance_matrix (numpy.ndarray): a kxk square input representing mutual distances
            between k points.
        eps (double): diffusion map parameter for K = exp( -distance_matrix ** 2 / (2 * eps) ).

    Returns:
        eigenvales (np.ndarray): a length k array of eigenvalues.
            eigenvectors (numpy.ndarray): a kxk array representing eigenvectors (descending order).
    '''
    K = np.exp(-distance_matrix**2/ (2. * eps) )
    d_K = np.squeeze(np.asarray(K.sum(axis = 1)))
    d_K_inv = np.power(d_K,-1)
    d_K_inv = np.nan_to_num(d_K_inv)
    L = d_K_inv*(d_K_inv*K).T
    d_L = np.squeeze(np.asarray(L.sum(axis = 1)))
    d_L_inv = np.power(d_L,-alpha)
    M = d_L_inv*(d_L_inv*L).T
    eigs = la.eigh(M)
    if eig_lower_bound is None:
        eig_lower_bound = 0
    if eig_upper_bound is None:
        eig_upper_bound = len(eigs[0])
    return (eigs[0][::-1][eig_lower_bound:eig_upper_bound],
            eigs[1].T[::-1].T[:,eig_lower_bound:eig_upper_bound])
