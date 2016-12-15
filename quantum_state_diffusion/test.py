import numpy as np
import sdeint

A = np.array([[0., -0.],
              [0., -0.]], dtype = np.complex128)

B = np.hstack([np.diag([1.0,1.0]),np.diag([1.0j,1.0j])]) # diagonal, so independent driving Wiener processes

tspan = np.linspace(0.0, 100.0, 10001)
x0 = np.array([0., 0.], dtype = np.complex128)

def f(x, t):
    return A.dot(x)

def G(x, t):
    return B

result = sdeint.itoint(f, G, x0, tspan)
print (result)
