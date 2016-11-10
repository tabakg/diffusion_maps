# Diffusion Maps
This python script uses several libraries to perform dimensionality reduction
using diffusion maps. The first part involves finding nearest neighbors,
which is done using the package vp_tree that can be generated from:
https://github.com/tabakg/cpp/tree/master/vp_trees_with_python_interface

I then generate some random data via jump trajectories
and follow the instructions for diffusion maps
as described by the paper "Diffusion Maps, Spectral Clustering and
Eigenfunctions of Fokker-Planck Operators" found on:
http://www.wisdom.weizmann.ac.il/~nadler/Publications/dm_nips05.pdf

## Quantum State Diffusion Trajectories
Another option for generating data is to sample quantum unravellings using
quantum state diffusion instead of jump trajectories. Although both methods
are guaranteed to produce the same density matrix in expectation, the
unravellings themselves may be different.  This is done in a separate
repository:
https://github.com/tabakg/quantum_state_diffusion
