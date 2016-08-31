# Diffusion Maps
This python script uses several libraries to perform dimensionality reduction
using diffusion maps. The first part involves finding nearest neighbors,
which is done using the package vp_tree that can be generated from:
https://github.com/tabakg/cpp/tree/master/vp_trees_with_python_interface

I then generate some random data and follow the instructions for diffusion maps
as described by the paper "Diffusion Maps, Spectral Clustering and
Eigenfunctions of Fokker-Planck Operators" found on:
http://www.wisdom.weizmann.ac.il/~nadler/Publications/dm_nips05.pdf
