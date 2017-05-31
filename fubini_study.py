import numpy as np
# cimport numpy as np

def inner_to_FS(z):
    if z >= 1.:
        return 0.
    else:
        return np.arccos(np.sqrt(z))

inner_to_FS_vec = np.vectorize(inner_to_FS)

def FS_metric(u, v):
    l = u.shape[-1] ## dimensionality = 2N

    if v.shape != u.shape:
        raise ValueError()
    if l%2 != 0:
        raise ValueError()

    if len(u.shape) == 1:
        num_points = 1
    else:
        num_points = u.shape[0]

    l = l/2
    u_r,u_i = np.split(u,2,axis=-1)
    v_r,v_i = np.split(v,2,axis=-1)

    inner = ( (np.dot(u_r,v_r.T) + np.dot(u_i,v_i.T))**2
            + (np.dot(u_r,v_i.T) - np.dot(u_i,v_r.T))**2  )

    return inner_to_FS_vec(inner)
