# Diffusion Maps
This python script uses several libraries to perform dimensionality reduction
using diffusion maps.
I then generate some random data via jump trajectories
and follow the instructions for diffusion maps
as described by the paper "Diffusion Maps, Spectral Clustering and
Eigenfunctions of Fokker-Planck Operators" found on:
http://www.wisdom.weizmann.ac.il/~nadler/Publications/dm_nips05.pdf


## Note:

In the future the installation requirements will be grouped inside of a Docker or Singularity file. We detail the installation instructions below.
 
## Installing Anaconda
I might make changes in the repos to make it easier to install and manage. In the meantime, here is how I install my packages.
 
After anaconda installation, we did the following:
 
Creating a new Python environment:
conda create -n yourenvname python=x.x anaconda
 
yourenvname is the name of the environment
x.x is the version (we made two of those, 3.5 and 2.7)
 
Go into environment (in terminal): source activate yourenvname
Exiting an environment: source deactivate
 
When you are inside an environment, running Python or installing packages works with that particular version of Python.
___________________________________________________
 
## Installing packages
To run jupyter notebook, go to the right directory (after activating the right environment) and type: jupyter notebook (or use the anaconda launch tool).
 
install packages using conda: conda install package1 package2 package3
Install packages using pip: pip install package4 package5
 
package1 package2 package3 package4 package5 are packages to be installed
 
Packages we installed from conda: numpy matplotlib sympy scipy
Packages we installed from pip: qutip QNET hmmlean
 
Create python 3.5 (with numpy matplotlib sympy scipy qutip QNET)
Create python 2.7 (with numpy matplotlib sympy scipy)
___________________________________________________
 
## Installing Boost
For the python 2, I also installed Boost to use the C++ library.
 
For Mac, use: brew install boost-python
(see also http://www.pyimagesearch.com/2015/04/27/installing-boost-and-boost-python-on-osx-with-homebrew/ )
 
 
Also available to download directly on:
 
https://sourceforge.net/projects/boost/files/boost/1.62.0/
 
Installation:
Source activate into desired environment. Open folder:
 
./bootstrap.sh
./b2
./b2 install
 
___________________________________________________
 
## Modified Qutip:
For some parts (specifically, running a “hybrid” simulation involving a c-number time series generated by an HMM fed into an open quantum system, as used in Cascading_kerr_cavities.ipynb), the qutip package had to be modified. Specifically, I used the “multiprocess” library instead of “multiprocessing” in parallel.py. Multiprocess can be installed using: pip install multiprocess. The reason this is necessary is that time dependent time series must be programmatically generated, and multiprocessing uses “pickle.py” to store functions. This library does not support function closures, making programmatic generation of time-dependent functions difficult.
 
___________________________________________________
 
 
 
## Current repos:
### Diffusion_maps:
Unless otherwise noted, run using Python 2.7.
 
First, generate trajectories. See make_quantum_jump_trajectories below.
 
To run the most recent analysis, use the notebooks:
 
	Build_markov_kerr.ipynb
	Build_markov_kerr_qubit.ipynb
	Build_markov_absorptive.ipynb
	Cascading_kerr_cavities.ipynb
 
	
### make_quantum_jump_trajectories (folder):
The Ipython notebooks here generate quantum jump trajectories. These are saved into a subfolder called “trajectories”
Run notebooks with Python 3.5.
Quantum_state_diffusion (folder)
Generates quantum state diffusion trajectories. Currently not used for other things, so skip if not necessary.
To install, sdeint. Again, this can be installed using python setup.py install. Install in Python 3.5
To do a local installation, activate the right environment, clone a repo, cd into it, and use: python setup.py install
Vp_trees_cpp (folder)
Implementation in C++ for vantage point trees. Used by diffusion maps
 
 
### Other repos (not currently using):
 
Vp_trees_cpp:
 
Runs with Python 2.7.
A package with C++ bindings using boost, used by diffusion maps (below).
Quantum_state_diffusion:
There are several options for running this script, including Docker, Singularity (container services), as well as local installation. The README is comprehensive. 
Runs in Python 3.5
 
To do a local installation, activate the right environment, clone a repo, cd into it, and use: python setup.py install
 
