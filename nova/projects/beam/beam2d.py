import numpy as np
import pylab as pl

EI = 10


# element stiffness matrix
def elem_k(dL):
    k = np.matrix(
        [
            [12, 6 * dL, -12, 6 * dL],
            [6 * dL, 4 * dL**2, -6 * dL, 2 * dL**2],
            [-12, -6 * dL, 12, -6 * dL],
            [6 * dL, 2 * dL**2, -6 * dL, 4 * dL**2],
        ]
    )
    return k


# [k][v] = [F]
# [k][v1,t1,v2,t2]' = [P1,M1,P2,M2]'


L = 10

Nel = 51
dL = L / Nel
n = 4 + 2 * (Nel - 1)
k = np.zeros((n, n))

for i in range(Nel):
    k[2 * i : 2 * i + 4, 2 * i : 2 * i + 4] += elem_k(dL)

# k = k[2:,2:] # fix end
for i in range(2):
    k = np.delete(k, (0, 1), i)
# k = np.delete(k,(0,1),1)

F = np.zeros((n - 2, 1))  # loading vector
P = [[0.3, -0.1], [1, -0.95], [0.5, 1.5]]
for p in P:
    n = int(np.ceil(p[0] * Nel) * 2) - 1
    F[n] = p[1]

vs = np.linalg.lstsq(k, F)[0] * dL**3 / EI
Fo = np.dot(k, np.reshape(vs, (len(vs), 1)))

pl.plot(vs[::2], "o-")
