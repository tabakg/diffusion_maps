import pickle
import numpy as np

def load_trajectory(Regime = "absorptive_bistable",
                    location = './make_quantum_jump_trajectories/trajectory_data/'):
    ## Loading trajectory data

    ## load trajectory data from file

    pkl_file = open(location + Regime +'.pkl', 'rb')
    pkl_dict = pickle.load(pkl_file)
    pkl_file.close()

    ## pre-process expectation values

    traj_expects = np.concatenate(pkl_dict['expects']).real.T

    ## some useful numbers

    Ntraj = pkl_dict['Ntraj']
    duration = pkl_dict['psis'].shape[1] ##pkl_dict['duration']

    ##  Extract data into points of format (psi.real,psi.imag) from all trajectories.

    traj_data = np.concatenate(
                [[ np.concatenate([f(pkl_dict['psis'][traj_num][time_num])
                    for f in (lambda x: x.real, lambda x: x.imag) ])
                        for traj_num in range(Ntraj)]
                            for time_num in range(int(duration))])
    return Ntraj,duration,traj_data,traj_expects
