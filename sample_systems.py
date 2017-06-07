from qnet.algebra.operator_algebra import *
from qnet.algebra.circuit_algebra import *
import qnet.algebra.state_algebra as sa
from qnet.circuit_components.displace_cc import Displace

from qnet.algebra.operator_algebra import get_coeffs ### used later to make time-dependent terms from reduced model.

from qutip import *
from scipy import *

import sympy; sympy.init_printing(use_latex="mathjax")
from sympy import sqrt

import numpy as np

from IPython.display import display

import pickle

# Define Kerr parameters
chi = symbols("chi", real=True, positive=True)
Delta = symbols("Delta", real=True)
kappa_1, kappa_2 = symbols("kappa_1, kappa_2", real=True, positive=True)
alpha0 = symbols("alpha_0")

def make_kerr_slh(index = 0, which_symbols = 'qnet',params = None):
    '''
    Make a kerr slh with given index. The constants chi, Delta, kappa_1 and kappa_2
    are also given an index.
    '''
    if which_symbols == 'qnet':
        a = Destroy(str(index))
        a.space.dimension = params['Nfock']
    elif which_symbols == 'sympy':
        a = symbols('a_'+str(index))
    else:
        raise ValueError('which_symbols must be qnet or sympy.')

    if params is None:
        chi_ = symbols("chi", real=True, positive=True)
        Delta_ = symbols("Delta", real=True)
        kappa_1_ = symbols('kappa_'+str(index)+'1', real = True, positive = True)
        kappa_2_ = symbols('kappa_'+str(index)+'2', real = True, positive = True)
    else:
        chi_ = params[chi]
        Delta_ = params[Delta]
        kappa_1_ = params[kappa_1]
        kappa_2_ = params[kappa_2]

    if which_symbols == 'qnet':
        S = -identity_matrix(2)
        L = [sqrt(kappa_1_)*a, sqrt(kappa_2_)*a]
        H = Delta_*a.dag()*a + chi_/2*a.dag()*a.dag()*a*a
        kerr_slh = SLH(S, L, H).toSLH()
        return kerr_slh, [a]

    elif which_symbols == 'sympy':

        N,x,p = symbols('N x p')
        a1 = (x+1j*p)/2.

        S = -identity_matrix(2)
        L = [np.sqrt(kappa_1_)*a1, np.sqrt(kappa_2_)*a1]
        H = Delta_*N + chi_/2*N*(N+1)
        kerr_slh = SLH(S, L, H).toSLH()

        return kerr_slh, [N,x,p]

def make_traj(slh,
               Tsim,
               obsq,
               ntraj = 1,
               seeds = [1],
             ):
    #### Running stochastic trajectory on given system
    H, L = slh.HL_to_qutip()
    psi0 = qutip.tensor(*[qutip.basis(d,0) for d in H.dims[0]])
    mcdata = qutip.mcsolve(H, psi0, Tsim, L,
                       obsq, ntraj=ntraj,
                       options=qutip.Odeoptions(store_states=True,average_expect=False, seeds = seeds))
    return mcdata
