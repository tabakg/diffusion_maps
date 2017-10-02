'''
Gil Tabak

Nov 3, 2016

This notebook uses the library sdeint to perform quantum state diffusion trajectories.
The inputs are purposely similar to qubit functions like mcsolve to make
integration easier later.

'''

import numpy as np
import numpy.linalg as la
import sdeint
from scipy import sparse
from time import time

from multiprocess import Pool

### Plotting
import matplotlib as mil
mil.use('TkAgg')
import matplotlib.pyplot as plt

class drift_diffusion_holder(object):
    '''
    We include a way to update L*psi and l = <psi,L,psi> when t changes.
    This makes the code somewhat more efficient since these values are used
    both for the drift f and the diffusion G terms, and don't have to be
    recomputed each time.
    '''
    def __init__(self, H, Ls, tspan, normalized_equation=True):
        self.t_old = min(tspan) - 1.
        self.H = H
        self.Ls = Ls
        self.Lpsis = None
        self.ls = None
        if normalized_equation:
            self.f = self.f_normalized
            self.G = self.G_normalized
        else:
            self.f = self.f_non_normalized
            self.G = self.G_non_normalized

    def update_Lpsis_and_ls(self, psi, t):
        '''Updates Lpsis and ls.

        If t is different than t_old, update Lpsis, ls, and t_old.
        Otherwise, do nothing.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float
        '''
        if t != self.t_old:
            self.Lpsis = [L.dot(psi) for L in self.Ls]
            self.ls = [Lpsi.dot(psi.conj()) for Lpsi in self.Lpsis]
            self.t_old = t

    def f_normalized(self, psi, t):
        '''Computes drift f.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: array of shape (d,)
            to define the deterministic part of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        return (-1j * self.H.dot(psi)
                - sum([ 0.5*(L.H.dot(Lpsi) + np.conj(l)*l*psi)
                - np.conj(l)*(Lpsi)
                    for L,l,Lpsi in zip(self.Ls, self.ls, self.Lpsis)]))

    def G_normalized(self, psi, t):
        '''Computes diffusion G.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: returning an array of shape (d, m)
            to define the noise coefficients of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        complex_noise = np.vstack([Lpsi - l*psi
                        for Lpsi, l in zip(self.Lpsis, self.ls)]) / np.sqrt(2.)
        return np.vstack([complex_noise.real, 1j*complex_noise.imag]).T

    def f_non_normalized(self, psi, t):
        '''Computes drift f.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: array of shape (d,)
            to define the deterministic part of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        return (-1j * self.H.dot(psi)
                - sum([ 0.5*(L.H.dot(Lpsi)) - np.conj(l)*(Lpsi)
                    for L,l,Lpsi in zip(self.Ls, self.ls, self.Lpsis)]))

    def G_non_normalized(self, psi, t):
        '''Computes diffusion G.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: returning an array of shape (d, m)
            to define the noise coefficients of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        complex_noise = np.vstack(self.Lpsis) / np.sqrt(2.)
        return np.vstack([complex_noise.real, 1j*complex_noise.imag]).T

def qsd_solve(H,
              psi0,
              tspan,
              Ls,
              sdeint_method,
              obsq=None,
              normalized_equation=True,
              normalize_state=True,
              ntraj=1,
              processes=8,
              seed=1):
    '''
    Args:
        H: NxN csr matrix, dtype = complex128
            Hamiltonian.
        psi0: Nx1 csr matrix, dtype = complex128
            input state.
        tspan: numpy array, dtype = float
            Time series of some length T.
        Ls: list of NxN csr matrices, dtype = complex128
            System-environment interaction terms (Lindblad terms).
        sdeint_method (Optional) SDE solver method:
            Which SDE solver to use. Default is sdeint.itoSRI2.
        obsq (optional): list of NxN csr matrices, dtype = complex128
            Observables for which to generate trajectory information.
            Default value is None (no observables).
        normalized_equation (optional): Boolean
            Use the normalized quantum state diffusion equations. (TODO: case False)
        normalize_state (optional): Boolean
            Whether to numerically normalize the equation at each step.
        ntraj (optional): int
            number of trajectories.
        processes (optional): int
            number of processes. If processes == 1, don't use multiprocessing.
        seed (optional): int
            Seed for random noise.

    Returns:
        A dictionary with the following keys and values:
            ['psis'] -> np.array with shape = (ntraj,T,N) and dtype = complex128
            ['obsq_expects'] -> np.array with shape = (ntraj,T,len(obsq)) and dtype = complex128

    '''

    ## Check dimensions of inputs. These should be consistent with qutip Qobj.data.
    N = psi0.shape[0]
    if psi0.shape[1] != 1:
        raise ValueError("psi0 should have dimensions Nx1.")
    a,b = H.shape
    if a != N or b != N:
        raise ValueError("H should have dimensions NxN (same size as psi0).")
    for L in Ls:
        a,b = L.shape
        if a != N or b != N:
            raise ValueError("Every L should have dimensions NxN (same size as psi0).")
    ## Determine seeds for the SDEs
    if type(seed) is list or type(seed) is tuple:
        assert len(seed) == ntraj
        seeds = seed
    elif type(seed) is int or seed is None:
        np.random.seed(seed)
        seeds = [np.random.randint(1000000) for _ in range(ntraj)]
    else:
        raise ValueError("Unknown seed type.")

    T_init = time()
    psi0_arr = np.asarray(psi0.todense()).T[0]
    drift_diffusion = drift_diffusion_holder(H, Ls, tspan)

    # '''Generate psis with single processing'''
    # psis = np.asarray([ sdeint_method(drift_diffusion.f, drift_diffusion.G,
    #    psi0_arr,tspan) for _ in range(ntraj)])

    '''Generate psis with multiprocessing'''
    def SDE_helper(args, s):
        '''Let's make different wiener increments for each trajectory'''
        m = 2 * len(Ls)
        N = len(tspan)-1
        h = (tspan[N-1] - tspan[0])/(N - 1)
        np.random.seed(s)
        dW = np.random.normal(0.0, np.sqrt(h), (N, m))
        return sdeint_method(*args, dW=dW, normalized=normalize_state)
    pool = Pool(processes=processes,)
    params = [[drift_diffusion.f, drift_diffusion.G, psi0_arr, tspan]] * ntraj
    psis = np.asarray(pool.map(lambda z: SDE_helper(z[0], z[1]),
        zip(params, seeds)))

    ## Obtaining expectations of observables
    obsq_expects = (np.asarray([[ np.asarray([ob.dot(psi).dot(psi.conj())
                        for ob in obsq])
                            for psi in psis[i] ] for i in range(ntraj)])
                                if not obsq is None else None)

    T_fin = time()
    print ("Run time:  ", T_fin - T_init, " seconds.")
    return {"psis":psis, "obsq_expects":obsq_expects, "seeds":seeds}

if __name__ == "__main__":

    psi0 = sparse.csr_matrix(([0,0,0,0,0,0,0,1.]),dtype=np.complex128).T
    H = sparse.csr_matrix(np.eye(8),dtype=np.complex128)
    Ls = [sparse.csr_matrix( np.diag([np.sqrt(i) for i in range(1,8)],k=1),dtype=np.complex128)]
    tspan = np.linspace(0, 10.0, 3000)
    obsq = [sparse.csr_matrix(np.diag([i for i in range(4)]*2), dtype=np.complex128)]

    ntraj = 10

    D = qsd_solve(H, psi0, tspan, Ls, sdeint.itoSRI2, obsq = obsq, ntraj = ntraj, normalized_equation=False, normalize_state=True)

    psis = D["psis"]
    obsq_expects = D["obsq_expects"]

    print ("Last point of traj 0: ", psis[0][-1])
    print ("Norm of last point in traj 0: ", la.norm(psis[0][-1]))  ## should be close to 1...

    for i in range(ntraj):
        plt.plot(tspan, obsq_expects[i,:,0].real,  linewidth=0.3)

    ave_traj = np.average(np.array([obsq_expects[i,:,0].real
        for i in range(ntraj)]), axis=0)
    plt.plot(tspan, ave_traj, color='k',  linewidth=2.0)
    plt.show()
