import sys
sys.path.append("/Users/gil/Documents/repos/c_pp_stuff/vp_trees_with_python_interface")
import vp_tree

import time

import numpy as np
from numpy import linalg as la
from numpy.random import normal
import random

import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs


printing_calculations = False

def make_gaussian_data(mu,sigma,num,dim):
    return (np.reshape(np.array([random.normalvariate(mu,sigma)
        for i in range(num*dim)]),(num,dim)) .tolist())

def run_diffusion_map(data, params, symmetric = False, eig_vec_both_sides = False, tol = 1e-14):
    '''
    Optional argument symmetric detemines whether to use the symmetrized M_s or the non-symmetrized version. 
    The symmetrized version could improve stability/performance, etc.
    
    Optional argument eig_vec_both_sides determines whether to compute both the left and right eigenvectors. If it is false, only     the right eigenvectors are computed.
    
    
    '''
    
    epsilon, gaussian_epsilon,alpha,data_size = params["epsilon"],params["gaussian_epsilon"],params["alpha"],params["data_size"]

    D = {tuple(value):i for i,value in enumerate(data)}

    t0 = time.time()
    tree = vp_tree.tree_container(data) ## make vp tree from data.
    t1 = time.time()
    s = tree.print_tree() ## string with tree contents.
    t2 = time.time()

    #### Diffusion Mapping.
    ## We use the default metric here.
    ## For the FS_metric, use: vp_tree.FS_metric(x,y).
    ## Remember to normalize x and y first!!
    ## print vp_tree.FS_metric((x/la.norm(x)).tolist(),(y/la.norm(y)).tolist())

    K = sparse.lil_matrix((data_size,data_size)) ## efficient format for constructing matrices.

    for i,x in enumerate(data):
        neighbors = tree.find_within_epsilon(x,epsilon)
        for y in neighbors:
            j = D[tuple(y)]
            K[i,j] = np.exp(-la.norm(np.asarray(x)-np.asarray(y))**2 / (2.0 * gaussian_epsilon))
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
    
    if symmetric:
        e_vals_tmp,e_vecs_tmp = eigsh(M, k = params["eigen_dims"], maxiter = data_size * 100 )
        ## change of basis below for right eigenvectors 
        e_vecs = np.asarray(D_s * np.asmatrix(e_vecs_tmp))
        e_vals,e_vecs = real_and_sorted(e_vals_tmp,e_vecs)
        if eig_vec_both_sides:
            ## change of basis below for left eigenvectors 
            e_vecs_left = np.asarray(D_s_inv * np.asmatrix(e_vecs_tmp) )
            _,e_vecs_left = real_and_sorted(e_vals_tmp,e_vecs_left)
    else: ## not symmetric
        e_vals,e_vecs = eigs(M, k = params["eigen_dims"], maxiter = data_size * 100 )
        e_vals,e_vecs = real_and_sorted(e_vals,e_vecs)
        if eig_vec_both_sides:
            e_vals_left,e_vecs_left = eigs(M.T, k = params["eigen_dims"], maxiter = data_size * 100 )
            e_vals_left,e_vecs_left = real_and_sorted(e_vals_left,e_vecs_left)
    
    t5 = time.time()

    if printing_calculations:
        print ("making the tree",t1-t0)
        print ("printing tree string (optional)",t2-t1)
        print ("Finding neighbors and generating K matrix", t3-t2)
        print ("calculations for M", t4-t3)
        print ("finding diffusion eigenvalues and eigenvectors",t5-t4)
    if eig_vec_both_sides:
        return e_vals,e_vecs,e_vecs_left
    else:
        return e_vals,e_vecs

def generate_data_n_gaussians(params):
    if params["data_size"] % params["n"] != 0:
        print "data size must be divisible by " + str(params["n"]) + " ."
        return
    num_per_well = params["data_size"] / params["n"] ## assume it's even for now
    l = []
    for mu in params['mus']:
        l += make_gaussian_data(mu,params["sigma"],num_per_well,params["dim"])
    return l