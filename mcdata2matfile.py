from scipy.io import savemat
from IPython.display import FileLink, display
import numpy as np

def mcdata2matfile(mcdata, file_name, obs, save_mat = True, save_pkl = False):
    """
    Takes an mcdata object and the observables and stores the states, expectations, times,
    observable labels (str and latex), random seeds, number of trajectories as:

        {
            "psis": psis,                          # shape (ntraj, ntimes, dim_psi) complex128
            "expects": expects,                    # shape (ntraj, ntimes, num_observables) complex128
            "times": times,                        # shape (ntimes) float64
            "observable_str": observable_str,      # shape (num_observables) string
            "observable_latex": observable_latex,  # shape (num_observables) string
            "seeds": seeds,                        # shape (ntraj) int
        }

    """
    ntraj = mcdata.ntraj
    assert ntraj >= 1

    psis = np.array([
            np.vstack([
                    mcdata.states[jj][tt].data.toarray().astype("complex128").ravel()
                    for tt in range(len(mcdata.times))])
            for jj in range(ntraj)])

    expects = np.array([np.array(mcdata.expect[jj]).astype("complex128").T for jj in range(ntraj)])
    times = mcdata.times
    observable_str = [str(o) for o in obs]
    observable_latex = [o._repr_latex_() for o in obs]
    seeds = mcdata.seeds
    mdict = {
            "psis": psis,
            "expects": expects,
            "times": times,
            "observable_str": observable_str,
            "observable_latex": observable_latex,
            "seeds": seeds,
    }
    if save_mat:
        savemat(file_name, mdict)
        display(FileLink(file_name+".mat"))
    if save_pkl:
        import pickle
        output = open(file_name + ".pkl", 'wb')
        pickle.dump( mdict, output)
        output.close()
        display(FileLink(file_name+".pkl"))
    return mdict
