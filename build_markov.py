################################################################################
### Imports
################################################################################

import os
import sys

#### VP Tree

sys.path.append(os.getcwd() + "/vp_trees_cpp/vp_trees_cpp")

from load_trajectory import load_trajectory
from fubini_study import FS_metric

## load trajectory data from file
import pickle

## diffusion maps
if sys.version[0] == '2':
    from vp_tree import FS_metric
    from vp_tree import tree_container
    from diffusion_maps import run_diffusion_map as diffusion_map
    from diffusion_maps import run_diffusion_map_dense as diffusion_map_dense
elif sys.version[0] == '3':
    from diffusion_maps_py3 import run_diffusion_map_dense as diffusion_map_dense

## numerical
import math
import numpy as np
from numpy import linalg as la
# from numpy.random import normal
import random
from scipy import sparse
# from scipy.sparse.linalg import eigsh
# from scipy.sparse.linalg import eigs

from sklearn.cluster import k_means
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from hmmlearn import hmm

## plotting libraries
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.cm as cm

from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

## Style for plots
plt.style.use("ggplot")

# 3D plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## videos

from IPython.display import HTML
from matplotlib import animation, rc

## misc
from bisect import bisect_left
import collections
from scipy.stats import rankdata
import time

################################################################################
#### Supporting Functions
################################################################################

def sorted_eigs(e_vals,e_vecs):
    ## Then sort the eigenvectors and eigenvalues s.t. the eigenvalues monotonically decrease.
    l = zip(e_vals,e_vecs.T)
    l = sorted(l,key = lambda z: -z[0])
    return np.asarray([el[0] for el in l]),np.asarray([el[1] for el in l]).T

def make_markov_model(labels,n_clusters):
    T = np.zeros((n_clusters,n_clusters))
    for i in range(len(labels)-1):
        j = i + 1
        T[labels[i],labels[j]] += 1
    row_norm = np.squeeze(np.asarray(T.sum(axis = 1)))
    row_norm = np.power(row_norm,-1)
    row_norm = np.nan_to_num(row_norm)
    return (T.T*row_norm).T

def next_state(state,T_cum):
    r = np.random.uniform()
    return bisect_left(np.asarray(T_cum)[state], r)

def run_markov_chain(start_cluster, T, steps = 10000):
    '''
    Run Markov chain for a single trajectory.
    '''

    T_cum = np.matrix(np.zeros_like(T))

    for i in range(T.shape[1]):
        s = 0.
        for j in range(T.shape[0]):
            s += T[i,j]
            T_cum[i,j] = s

    row_sums = np.asarray([sum(T[i,:]) for i in range(T.shape[1])])
    for row,row_sum in enumerate(row_sums):
        '''
        The row sums can sometimes end up not being 1.
        Assuming they are >=0, we fix that here.
        '''
        if row_sum == 0.:
            T_cum[row] = 1.
        if row_sum != 1.:
            T_cum[row] /= row_sum

    current = start_cluster
    outs = []
    for i in range(steps):
        outs.append(current)
        current = next_state(current,T_cum)
    return np.asarray(outs)

def get_cluster_labels(X,n_clusters = 30, num_nearest_neighbors = 100):
    knn_graph = kneighbors_graph(X, num_nearest_neighbors, include_self=False)
    model = AgglomerativeClustering(linkage='ward',
                                    connectivity=knn_graph,
                                    n_clusters=n_clusters)
    model.fit(X)
    labels = model.labels_
    return labels

def get_clusters(labels,n_clusters = 30):
    clusters = [[] for _ in range(n_clusters)]
    for point,cluster in enumerate(labels):
        clusters[cluster].append(point)
    return clusters

def get_hmm_hidden_states(X,
                            n_clusters = 30,
                            return_model = False,
                            n_iter=100,
                            tol=1e-2,
                            covariance_type = 'full',
                            random_state = 1,
                            verbose = False,
                            Ntraj = None):
    if Ntraj is None:
        Ntraj = 1
    assert X.shape[0] % Ntraj == 0
    lengths = [X.shape[0] / Ntraj] * Ntraj
    hmm_model = hmm.GaussianHMM(n_components=n_clusters, covariance_type=covariance_type, n_iter = n_iter, tol = tol, random_state = random_state)
    hmm_model.fit(X,lengths)
    if verbose:
        print("converged", hmm_model.monitor_.converged)
    hidden_states = hmm_model.predict(X)
    if not return_model:
        return hidden_states
    else:
        return hidden_states, hmm_model

def get_obs_sample_points(obs_indices,traj_expects):
    '''
    Observed quantities at sampled points
    '''
    return np.asarray([traj_expects[:,l] for l in obs_indices])

def get_expect_in_clusters(obs_indices,clusters, n_clusters, obs_sample_points):
    '''
    Get the average expectation value in each cluster.
    '''
    expect_in_clusters = {}
    for l in obs_indices:
        expect_in_clusters[l] = [0. for _ in range(n_clusters)]
        for clust_index,cluster in enumerate(clusters):
            for point in cluster:
                expect_in_clusters[l][clust_index] += obs_sample_points[l][point]
            if len(cluster) != 0:
                expect_in_clusters[l][clust_index] /= float(len(cluster))
            else:
                expect_in_clusters[l][clust_index] = None
    return expect_in_clusters

def get_obs_generated(  obs_indices,
                        T_matrix, ## Transition matrix used
                        expect_in_clusters,
                        steps = 10000,
                        n_clusters = 10,
                        start_cluster = 0, ## index of starting cluster
                     ):

    steps = run_markov_chain(start_cluster,T_matrix, steps = steps)
    obs_generated = np.asarray([[expect_in_clusters[l][cluster] for cluster in steps ] for l in obs_indices])
    return obs_generated

def get_reduced_model_time_series(expect_in_clusters,indices,point_in_which_cluster):
    '''
    Return the average expectation value over clusters
    as a time series of the true trajectory
    '''
    return [[expect_in_clusters[l][point_in_which_cluster[point]]
                for point in range(len(indices))] for l in obs_indices]

################################################################################
#### Plotting Functions
################################################################################

def contour_plot(Mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(abs(Mat), interpolation='nearest')
    fig.colorbar(cax)
    plt.show()

def make_density_scatterplts(X,Y,label_shift = 0):
    fig, ax = plt.subplots(nrows=X.shape[-1],ncols=Y.shape[-1],figsize = (Y.shape[-1]*10,X.shape[-1]*10))
    for i,row in enumerate(ax):
        for j,col in enumerate(row):
            col.set_title( "O" + str(j+1) +" vs  Phi" + str(i+1+label_shift))
            x = X[:,i]
            y = Y[:,j]

            xy = np.vstack([x,y])
            z = gaussian_kde(xy)(xy)
            col.scatter(x, y, c=np.log(z), s=5, edgecolor='')
    plt.show()

def ellipses_plot(X,indices,hmm_model,n_clusters,std_dev = 1):
    # Calculate the point density
    ### from http://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    x = X[:,indices[0]]
    y = X[:,indices[1]]

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    covariances = np.asarray([[[hmm_model.covars_[clus,n,m]
                                for n in indices]
                                    for m in indices]
                                        for clus in range(n_clusters)])
    means = np.asarray([[hmm_model.means_[clus,n]
                                for n in indices]
                                    for clus in range(n_clusters)])

    sorted_eig_lst = [sorted_eigs(*la.eig(cov)) for cov in covariances]
    angles = np.rad2deg(np.asarray([np.arctan2(*v[1][:,0]) for v in sorted_eig_lst]))
    widths = [2*std_dev*np.sqrt(v[0][0]) for v in sorted_eig_lst]
    heights = [2*std_dev*np.sqrt(v[0][1]) for v in sorted_eig_lst]

    es = [Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor='black', facecolor='none',linewidth = 1)
             for mean,width,height,angle in zip(means,widths,heights,angles) ]

    fig = plt.figure(0,figsize= (10,10))
    ax = fig.add_subplot(111, )
    ax.set_title("coordinates "+str(indices[0]+1)  +","+str(indices[1]+1) )

    for e in es:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_clip_box(ax.bbox)

    # ax.set_xlim(-.06, .06 )
    # ax.set_ylim(-.06,.06 )
    ax.scatter(x, y, c=np.log(z), s=100, edgecolor='')

################################################################################
#### Testing Transition Times
################################################################################

### Currently in notebook

################################################################################
#### Object used to hold the model
################################################################################

class dim_red_builder:
    def __init__(self,
                Regime= "kerr_bistable",
                sample_type = 'last_n',
                num_sample_points = 1000,
                hand_picked_indices = None,
                obs_indices = None,
                name = None,
                mcdata = None,
                ):
        '''

        '''
        self.Regime = Regime ## used for file naming

        if name is None:
            self.name = Regime
        else:
            self.name = name

        if mcdata is None:
            self.Ntraj, self.duration, self.traj_data, self.traj_expects = load_trajectory(self.Regime)
        else:
            self.Ntraj, self.duration, states, self.traj_expects = mcdata.ntraj, mcdata.times.shape[0], mcdata.states, np.concatenate(mcdata.expect,axis=1)
            self.traj_data = np.concatenate(
                            [[ np.concatenate([f(states[traj_num][time_num].data.todense())
                                for f in (lambda x: x.real, lambda x: x.imag) ])
                                    for traj_num in range(self.Ntraj)]
                                        for time_num in range(int(self.duration))])

        self.num_data_points = len(self.traj_data)
        self.num_sample_points = num_sample_points ## NA if using hand_picked indices
        self.sample_type = sample_type

        ### Which observables to use
        if obs_indices is None:
            self.obs_indices = range(self.traj_expects.shape[0])
        else:
            self.obs_indices = obs_indices

        if sample_type == 'uniform_time':
            downsample_rate = num_data_points / num_sample_points
            self.sample_indices = range(num_data_points)[::downsample_rate]
        elif sample_type == 'uniform_random':
            self.sample_indices = random.sample(range(num_data_points),num_sample_points)
        elif sample_type == 'first_n':
            self.sample_indices = range(num_sample_points)
        elif sample_type == 'last_n':
            self.sample_indices = range(self.num_data_points - num_sample_points, self.num_data_points)
        elif sample_type == 'hand_picked':
            if hand_picked_indices is None:
                raise ValueError("if using sample type hand_picked indices, must specify indices with hand_picked_indices")
            self.sample_indices = hand_picked_indices
        elif sample_type == 'all':
            self.sample_indices = range(self.num_data_points)
        else:
            raise ValueError("unknown sample_type")

        self.points = np.asarray([self.traj_data[i] for i in self.sample_indices])
        self.expects_sampled = np.asarray([[self.traj_expects[expects_index][i]
                                                for expects_index in self.obs_indices]
                                                    for i in self.sample_indices])
        self.status = 'not attempted'

    def run_diffusion_map(  self,
                            which_points = 'all', ## 'all', 'num_neighbors', or 'epsilon_cutoff'
                            load_diff_coords = False, ## Try to load file. If not found, run diffuion maps again
                            save_diff_coords = False, ## Save diffuion maps data into file. Overwrite if it exists.:
                            eps = 0.1,
                            alpha = 0.5, ### spectral parameter
                            eig_lower_bound = 0,
                            eig_upper_bound = 7,
                            num_neighbors = 300,  ## only used if which_points == 'num_neighbors'
                            epsilon_cutoff = 0.5, ## only used if which_points == 'epsilon_cutoff'
                         ):
        ########################################################################
        ### Attempting to load from file
        ########################################################################
        if load_diff_coords:
            try:
                pkl_file = open('diff_coords' + self.Regime +'.pkl', 'rb')
                self.vals,self.vecs = pickle.load(pkl_file)
                pkl_file.close()
            except IOError:
                print ("diffusion coordinates file not found. Instead running diffuions maps (this may take a few mintues).")
                load_diff_coords = False
        ########################################################################
        ### Not loading -> run diffusion maps
        ########################################################################
        if not load_diff_coords: ## if we're not loading,
            if which_points == 'all':
                distance_matrix = FS_metric(self.points,self.points)
                self.vals,self.vecs = diffusion_map_dense(
                                            distance_matrix,
                                            eps=eps,
                                            alpha = alpha,
                                            eig_lower_bound = eig_lower_bound,
                                            eig_upper_bound = eig_upper_bound,
                                            )
            elif which_points == 'num_neighbors' or which_points == 'epsilon_cutoff':
                ## If not using all neighbors, use the specialized nearest neighbors algorithm
                assert sys.version[0] == '2' ## these will work only in python 2 right now...
                diffusion_params = {"gaussian_epsilon" : eps,                   ## width of Gaussian kernel.
                                    "epsilon_cutoff" : epsilon_cutoff,          ## only used if which_points == 'epsilon_cutoff'
                                    "num_neighbors" : num_neighbors,            ## cutoff of number of neighbors, only if which_points == 'num_neighbors'
                                    "alpha" : alpha,                            ## coefficient to use for diffusion maps. See the wikipedia article
                                    "data_size" : len(self.points),                  ## total number of points
                                    "eigen_dims" :  eig_upper_bound,            ## number of lower dimensions to consider, i.e. number of eigenvectors to find.
                                    }
                self.vals, self.vecs = diffusion_map(
                                   self.points.tolist(),
                                   diffusion_params,
                                   symmetric=True,
                                   neighbor_selection = which_points)
            else:
                raise ValueError("Uknown value for which_points. Please use 'all', 'num_neighbors', or 'epsilon_cutoff'.")
            self.X = self.vecs[:,1:]
            self.status = 'success'
        ########################################################################
        ### Saving
        ########################################################################
        if save_diff_coords:
            pkl_file = open('diff_coords' + self.Regime +'.pkl', 'wb')
            pickle.dump((self.vals,self.vecs),pkl_file)
            pkl_file.close()
        return

    def load(self):
        f = open(self.name,'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self):
        f = open(self.name,'wb')
        pickle.dump(self.__dict__,f,2)
        f.close()

    def plot_obs_v_diffusion(self):
        make_density_scatterplts(self.X[:,0:],
            self.expects_sampled,label_shift = 0)

    def plot_diffusion_v_diffusion( self,
                                    color_by_percentile = True,
                                    max_coord1 = 4,
                                    max_coord2 = 4,
                                    ):
        for l in self.obs_indices:
            fig = plt.figure(figsize=(max_coord2*10,max_coord1*10))
            if color_by_percentile:
                expects_sampled_percentile = rankdata(self.expects_sampled[:,l], "average") / self.num_sample_points
            for k in range(max_coord1):
                for i in range(k+1,max_coord2):
                    ax = fig.add_subplot(max_coord1, max_coord2, k*max_coord2+i+1)
                    plt.scatter(self.X[:,k],self.X[:,i], c = expects_sampled_percentile)
                    plt.title("Observable" + str(l) + "; coordinates: " + str(i) + "versus " + str(k) )
            plt.show()

    def plot_diffusion_3d(self,
                            color_by_percentile = True,
                            coords = [0,1,2],
                            ):
        if len(coords) != 3:
            raise ValueError("number of coordinates must be 3 for 3D plot.")

        num_obs = len(self.obs_indices)
        fig = plt.figure(figsize=(num_obs*10,10))

        for l in self.obs_indices:
            ax = fig.add_subplot(1, num_obs, l+1, projection='3d')
            expects_sampled_percentile = rankdata(self.expects_sampled[:,l], "average") / self.num_sample_points
            ax.scatter(self.X[:,coords[0]],self.X[:,coords[1]],self.X[:,coords[2]],
                        c = expects_sampled_percentile)
        plt.show()

class markov_model_builder:
    def __init__(self, dim_red, name = None):
        try:
            self.X = dim_red.X
        except:
            print('Warning: did not find reduced coordinates X in dim_red')
        self.Ntraj = dim_red.Ntraj
        self.expects_sampled = dim_red.expects_sampled
        self.obs_indices = dim_red.obs_indices
        if name is None:
            self.name = dim_red.name + "_markov_builder"
        else:
            self.name = name
        self.status = 'not attempted'

    def load(self):
        f = open(self.name,'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self):
        f = open(self.name,'wb')
        pickle.dump(self.__dict__,f,2)
        f.close()

    def build_model(self,
                    n_clusters = 10,
                    method = 'hmm',
                    n_iter=1000,
                    covariance_type = 'full',
                    num_diff_coords = None,
                    tol=1e-2,
                    get_expects = True,
                    random_state = 1,
                    verbose = True,
                    which_coords = 'X',
                    ):
        '''
        method can be 'hmm' or 'agg_clustering'.

        which_coords can be 'X' for reduced coordiantes (e.g. diffusion coords)
        or 'expects' for expectation values
        '''
        if method == 'hmm' or method == 'agg_clustering':
            self.method = method
        else:
            raise ValueError("Unknown method type. method can be 'hmm' or 'agg_clustering'. ")
        self.n_clusters = n_clusters

        if which_coords == 'X':
            assert hasattr(self, 'X')
            if num_diff_coords is None:
                self.num_diff_coords = self.X.shape[-1]
            else:
                self.num_diff_coords = num_diff_coords
            X_to_use = self.X[:,:self.num_diff_coords]
        else:
            assert which_coords == 'expects'
            X_to_use = self.expects_sampled
        if self.method == 'hmm':
            # try:
            self.labels, self.hmm_model = get_hmm_hidden_states(X_to_use,
                                                            self.n_clusters,
                                                            return_model=True,
                                                            n_iter=n_iter,
                                                            tol=tol,
                                                            covariance_type = covariance_type,
                                                            random_state=random_state,
                                                            verbose = True,
                                                            Ntraj = self.Ntraj,
                                                            )
            # except:
                # self.status = 'failed to build hmm'
                # print('failed to build hmm')
                # raise ValueError("hmm failed.")
            self.clusters = get_clusters(self.labels,self.n_clusters)
            self.T = make_markov_model(self.labels,self.n_clusters)
            self.status = 'model built'
        elif self.method == 'agg_clustering':
            self.labels = get_cluster_labels(X_to_use, self.n_clusters)
            self.clusters = get_clusters(self.labels,self.n_clusters)
            self.T = make_markov_model(self.labels,self.n_clusters)
            self.status = 'model built'
        else:
            raise ValueError("Unknown method type. method can be 'hmm' or 'agg_clustering'. ")

        if get_expects:
            self.obs_sample_points = get_obs_sample_points(self.obs_indices,self.expects_sampled)
            self.expects_in_clusters = get_expect_in_clusters(self.obs_indices,self.clusters, self.n_clusters, self.obs_sample_points)

    def get_ordering_by_obs(self,obs_index = 0):
        assert self.status == 'model built'
        obs_used = self.expects_sampled[:,obs_index]
        expects_in_clusters = [np.average([obs_used[i] for i in self.clusters[k]]) for k in range(self.n_clusters) ]
        D = {num:i for num,i in  zip(expects_in_clusters,range(self.n_clusters))}
        cluster_order = [D[key] for key in sorted(D.keys())]
        return cluster_order

    def plot_transition_matrix(self, obs_index = None,):
        assert self.status == 'model built'
        if obs_index is None:
            contour_plot(self.T)
        elif isinstance(obs_index,int):
            cluster_order = self.get_ordering_by_obs(obs_index)
            def order_indices(mat):
                return np.asmatrix([[mat[i,j] for i in cluster_order] for j in cluster_order])
            contour_plot(order_indices(self.T))
        else:
            raise ValueError("obs_index should be None or an integer (representing an observable).")

    def ellipses_plot(self,indices = [0,1]):
        '''
        Works only for hmm, using the diffusion coordinates X.
        '''
        assert self.status == 'model built'
        ellipses_plot(self.X[:,:self.num_diff_coords],indices,self.hmm_model,self.n_clusters)

    def generate_obs_traj(self,steps = 10000,random_state = 1,start_cluster=0 ):
        assert self.status == 'model built'
        np.random.seed(random_state)
        return get_obs_generated(   self.obs_indices,
                                    self.T, ## Transition matrix used
                                    self.expects_in_clusters,
                                    steps = steps,
                                    n_clusters = self.n_clusters,
                                    start_cluster = start_cluster, ## index of starting cluster
                                )
