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



dim = 5
data_size = 20

mu = 0.0 ## data distribution mean.
sigma = 1.0 ## data distribution standard deviation.
epsilon = 4.0 ## cutoff for nearest neighbors.
gaussian_epsilon = epsilon / 3. ## width of Gaussian kernel.
alpha = 0.5 ## coefficient to use for diffusion maps. See the wikipedia article.

printing_calculations = False

print("Data are " + str(data_size) + " points of dimension " + str(dim) + " .")
print("Gaussian points with mu = " + str(mu) + ", sigma = "+str(sigma)+"; cutoff epsilon = "+str(epsilon)+" \n")

data = np.reshape(np.array([random.normalvariate(mu,sigma) for i in range(data_size*dim)]),(data_size,dim) ).tolist()

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

print "total neighbors found = ", len(K.nonzero()[0])

t3 = time.time()

d_X = np.squeeze(np.asarray(K.sum(axis = 1)))
if printing_calculations: print "d_X = ", d_X

K_coo = sparse.coo_matrix(K) ## can iterate over (i,j,v) of coo_matrix
L = sparse.lil_matrix((data_size,data_size))
for i,j,v in zip(K_coo.row, K_coo.col, K_coo.data):
    L[i,j] = v * np.power(d_X[i] * d_X[j], -alpha)
if printing_calculations: print "L = ",L.todense()

# ## Calculate transition matrix. Optional.
# P = sparse.lil_matrix((data_size,data_size))
# for i,j,v in zip(K_coo.row, K_coo.col, K_coo.data):
#     P[i,j] = v / d_X[i]
# if printing_calculations: print "P = ",L.todense()

D_no_power = np.squeeze(np.asarray(L.sum(axis = 1))) ## get diagonal elements of D
D_inv = np.power(D_no_power, -1.0)
D_inv = sparse.diags([D_inv],[0],format = 'csr')
M = D_inv * L.tocsr()

D_diag = np.power(np.squeeze(np.asarray(L.sum(axis = 1))),-0.5) ## get diagonal elements of D
D_s = sparse.diags([D_diag],[0],format = 'csr')
if printing_calculations: print "D_s = ", D_s.todense()

M_s = D_s * L.tocsr() * D_s
if printing_calculations: print M_s.todense()

t4 = time.time()

e_vals,e_vecs = eigsh(M_s, k = 10,maxiter = data_size * 100 )

# ## Fnd biggest eigenvector for P, to compare to phi_0. Optional.
# P_e_vals,P_e_vecs = eigs(P.T, k = 1,maxiter = data_size * 100 )

def normalize(v):
    return v / la.norm(v)

# phi_0 = d_X / sum(d_X)
# if printing_calculations: print normalize(phi_0)
# first_vec = P_e_vecs [:,0]
# if printing_calculations: print normalize (first_vec)

t5 = time.time()

print ("making the tree",t1-t0)
print ("printing tree string (optional)",t2-t1)
print ("Finding neighbors and generating K matrix", t3-t2)
print ("calculations for M_s", t4-t3)
print ("finding diffusion eigenvalues and eigenvectors",t5-t4)
