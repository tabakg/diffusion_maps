## local path to vp_tree library.
## Available at https://github.com/tabakg/vp_trees_cpp

import os
import sys

sys.path.append(os.getcwd() + "/vp_trees_cpp/vp_trees_cpp")

import vp_tree
import time

import numpy as np
from numpy import linalg as la
from numpy.random import normal
import random

import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

import warnings
printing_calculations = False

def make_gaussian_data(mu,sigma,num,dim):
    return (np.reshape(np.array([random.normalvariate(mu,sigma)
        for i in range(num*dim)]),(num,dim)) .tolist())

def run_diffusion_map(data, params, symmetric = False, metric = "euclidean", neighbor_selection = "epsilon_cutoff", eig_vec_both_sides = False, return_eigenvectors = True):
    '''
    data (2D numpy array, list of arrays, array of lists, or list of lists):
        input data points. Should all be the same size.

    params: A dictionary with some numerical parameters:
        "gaussian_epsilon": Bandwidth parameter eps s.t.
            K = (-d^2(x,y) / (2 * eps) )
        "alpha": scaling of local density.
            0: No scaling; 0.5: Fokker-Plank scaling; 1.: Full scaling
        "eigen_dims": number of lower dimensions k to find
            (including the steady-state \psi_0)
        "epsilon_cutoff": IF neighbor_selection = "epsilon_cutoff"
            Max distance to use for ignoring points,
        "num_neighbors": IF neighbor_selection == "num_neighbors"
            Max number of neighbors to use before ignoring far away points

    symmetric (Optional) Boolean:
        detemines whether to use the symmetrized M_s or the non-symmetrized
            version.
        The symmetrized version could improve stability/performance, etc.
            but requires matrix multiplication.

    neighbor_selection (Optional) String:
        "epsilon_cutoff" -- using some fixed epsilon.
        "num_neighbors" -- fixed number of neighbors.

    eig_vec_both_sides (Optional) Boolean:
        determines whether to compute both the left and right eigenvectors.
        If it is false, only  the right eigenvectors are computed.
        This is ignored if return_eigenvectors == False.
    return_eigenvectors (Optional) Boolean:
        Return (left) eigenvectors or not.
    '''

    ## process input to make sure it is a list of lists of doubles.
    ## This should work if the input is a list of lists, an array of lists,
    ## of a list of arrays.

    if len(data) == 0:
        raise ValueError("input data is empty.")
    if not isinstance(data,list):
        data = data.tolist()
    for i in range(len(data)):
        if not isinstance(data[i],list):
            data[i] = data[i].tolist()

    gaussian_epsilon,alpha,k = params["gaussian_epsilon"],params["alpha"],params["eigen_dims"]
    if neighbor_selection == "epsilon_cutoff":
        epsilon = params["epsilon"]
    elif neighbor_selection == "num_neighbors":
        num_neighbors = params["num_neighbors"]
    else:
        raise ValueError("Unknown neighbor selection method. Please use epsilon_cutoff or num_neighbors")

    data_size = len(data)

    if k > data_size - 2:
        warnings.warn("Low dimension k = params['eigen_dims'] is greater than number of points N - 2. Setting k = N - 2.")
        k = data_size - 2
    if k < 1:
        warnings.warn("k < 1 (possibly after being modified to N-2), returning empty list.")
        return []


    D = {tuple(value):i for i,value in enumerate(data)}

    t0 = time.time()
    tree = vp_tree.tree_container(data,metric) ## make vp tree from data.
    t1 = time.time()
    s = tree.print_tree() ## string with tree contents.
    t2 = time.time()

    #### Diffusion Mapping.

    def fill_K_row(i,x,neighbors,K,metric='euclidean'):
        '''
        Fill the ith row of the K similarity matrix,
        corresponding to point x with its neighbors.
        '''
        if metric == 'euclidean':
            for y in neighbors:
                j = D[tuple(y)] ## index of point y
                K[i,j] = np.exp(-la.norm(np.asarray(x)-np.asarray(y))**2 / (2.0 * gaussian_epsilon))
        elif metric == 'FS_metric':
            for y in neighbors:
                j = D[tuple(y)] ## index of point y
                K[i,j] = np.exp(-tree.FS_metric(x,y)**2 / (2.0 * gaussian_epsilon))
        return

    K = sparse.lil_matrix((data_size,data_size)) ## efficient format for constructing matrices.
    if neighbor_selection == "epsilon_cutoff":
        for i,x in enumerate(data):
            neighbors = tree.find_within_epsilon(x,epsilon,metric)
            fill_K_row(i,x,neighbors,K)
    elif neighbor_selection == "num_neighbors":
        print "using num_neighbors  "
        all_neighbors = tree.find_all_N_neighbors(num_neighbors, metric)
        for i,neighbors in enumerate(all_neighbors):
            x = data[i]
            fill_K_row(i,x,neighbors,K)

    if printing_calculations: print "K = ", K.todense()

    if printing_calculations: print "total neighbors found = ", len(K.nonzero()[0])

    t3 = time.time()

    d_X = np.squeeze(np.asarray(K.sum(axis = 1)))
    if printing_calculations: print "d_X = ", d_X

    K_coo = sparse.coo_matrix(K) ## can iterate over (i,j,v) of coo_matrix
    L = sparse.lil_matrix((data_size,data_size))
    for i,j,v in zip(K_coo.row, K_coo.col, K_coo.data):
        L[i,j] = v * np.power(d_X[i] * d_X[j], -alpha)
    if printing_calculations: print "L = ",L.todense()

    D_no_power = np.squeeze(np.asarray(L.sum(axis = 1))) ## get diagonal elements of D

    ## symmetrized
    if symmetric:
        D_frac_pow = np.asarray([np.power(d, - 0.5) if not d == 0. else 0. for d in D_no_power])
        D_s = sparse.diags([D_frac_pow],[0],format = 'csr')
        if eig_vec_both_sides: ## This matrix gives the change of basis for the left eigenvectors.
            D_frac_pow_inv = np.asarray([np.power(d, 0.5) if not d == 0. else 0. for d in D_no_power])
            D_s_inv = sparse.diags([D_frac_pow_inv],[0],format = 'csr')
        if printing_calculations:
            print "D_s = ", D_s.todense()
        M = D_s * L.tocsr() * D_s
        if printing_calculations:
            print M.todense()
    else: ## non-symmetrized. Must use eigensolver that doesn't assume symmetry.
        D_frac_pow = np.asarray([np.power(d, - 1) if not d == 0. else 0. for d in D_no_power])
        D_s = sparse.diags([D_frac_pow],[0],format = 'csr')
        M = D_s * L.tocsr()

    t4 = time.time()

    def real_and_sorted(e_vals,e_vecs):
        ## get real part, there should not be imaginary part.
        ## Then sort the eigenvectors and eigenvalues s.t. the eigenvalues monotonically decrease.
        e_vals,e_vecs = e_vals.real,e_vecs.real
        l = zip(e_vals,e_vecs.T)
        l.sort(key = lambda z: -z[0])
        return np.asarray([el[0] for el in l]),np.asarray([el[1] for el in l]).T

    if return_eigenvectors:
        if symmetric:
            e_vals_tmp,e_vecs_tmp = eigsh(M, k = k, maxiter = data_size * 100, return_eigenvectors = return_eigenvectors )
            ## change of basis below for right eigenvectors
            e_vecs = np.asarray(D_s * np.asmatrix(e_vecs_tmp))
            e_vals,e_vecs = real_and_sorted(e_vals_tmp,e_vecs)
            if eig_vec_both_sides:
                ## change of basis below for left eigenvectors
                e_vecs_left = np.asarray(D_s_inv * np.asmatrix(e_vecs_tmp) )
                _,e_vecs_left = real_and_sorted(e_vals_tmp,e_vecs_left)
        else: ## not symmetric
            e_vals,e_vecs = eigs(M, k = k, maxiter = data_size * 100,return_eigenvectors = return_eigenvectors )
            e_vals,e_vecs = real_and_sorted(e_vals,e_vecs)
            if eig_vec_both_sides:
                e_vals_left,e_vecs_left = eigs(M.T, k = k, maxiter = data_size * 100, return_eigenvectors = return_eigenvectors )
                e_vals_left,e_vecs_left = real_and_sorted(e_vals_left,e_vecs_left)
    else:
        if symmetric:
            e_vals = eigsh(M, k = k, maxiter = data_size * 100, return_eigenvectors = return_eigenvectors )
        else: ## not symmetric
            e_vals = eigs(M, k = k, maxiter = data_size * 100,return_eigenvectors = return_eigenvectors )
        e_vals = e_vals.real.tolist()
        e_vals.sort(key = lambda z: -z)

    t5 = time.time()

    if printing_calculations:
        print ("making the tree",t1-t0)
        print ("printing tree string (optional)",t2-t1)
        print ("Finding neighbors and generating K matrix", t3-t2)
        print ("calculations for M", t4-t3)
        print ("finding diffusion eigenvalues and eigenvectors",t5-t4)
    if return_eigenvectors == False:
        return e_vals
    elif eig_vec_both_sides:
        return e_vals,e_vecs,e_vecs_left
    else:
        return e_vals,e_vecs
    
def run_diffusion_map_dense(distance_matrix,eps = 0.2):
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
    d_L_inv = np.power(d_L,-0.5)
    M = d_L_inv*(d_L_inv*L).T
    eigs = la.eigh(M)
    return (eigs[0][::-1], eigs[1].T[::-1].T)

def generate_data_n_gaussians(params):
    if params["data_size"] % params["n"] != 0:
        print "data size must be divisible by " + str(params["n"]) + " ."
        return
    num_per_well = params["data_size"] / params["n"] ## assume it's even for now
    l = []
    for mu in params['mus']:
        l += make_gaussian_data(mu,params["sigma"],num_per_well,params["dim"])
    return l
