import cupy as cp
import numpy as np


'''
@numba.cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B."""
    i, j = numba.cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            if B[k, j] == 0:
                continue
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


TPB = 16

@numba.cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = numba.cuda.shared.array(shape=(TPB, TPB), dtype=numba.float32)
    sB = numba.cuda.shared.array(shape=(TPB, TPB), dtype=numba.float32)

    x, y = numba.cuda.grid(2)

    tx = numba.cuda.threadIdx.x
    ty = numba.cuda.threadIdx.y
    bpg = numba.cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        numba.cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        numba.cuda.syncthreads()

    C[x, y] = tmp
'''

if __name__ == "__main__":
    M = 65**2
    N = 1000

    a = np.ones((M, N), cp.float32)
    b = np.ones((N, 1), cp.float32)
    c = np.empty((M, 1), cp.float32)

    """
    _a = numba.cuda.to_device(a)
    _b = numba.cuda.to_device(b)
    _c = numba.cuda.to_device(c)
    """

    _a = cp.asarray(a)
    _b = cp.asarray(b)
    _c = cp.asarray(c)
    cp.matmul(_a, _b, _c)

    # matmul[16, 16](_a, _b, _c)
