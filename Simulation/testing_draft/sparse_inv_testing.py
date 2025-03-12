import taichi as ti
import numpy as np
import concurrent.futures

ti.init(arch=ti.cpu)

# Dimension of the matrix
n = 2048

# Build a sparse matrix using Taichi's SparseMatrixBuilder
A_builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100000, dtype=ti.f32)
@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), interval: ti.i32):
    for i in range(n):
        A[i, i] += n
        if i % interval == 0:
            A[i, i] += 1.0
fill(A_builder, 3)
A = A_builder.build()

# Setup a sparse solver (using LU factorization)
solver = ti.linalg.SparseSolver(solver_type="LU")
solver.compute(A)

# Prepare an array to store the inverse (dense representation)
A_inv = np.zeros((n, n), dtype=np.float32)

# Function to solve for one column of the inverse
def solve_column(i):
    b = np.zeros(n, dtype=np.float32)
    b[i] = 1.0  # i-th canonical basis vector
    x = solver.solve(b)
    return i, x

# Use ThreadPoolExecutor to parallelize the column solves
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(solve_column, i) for i in range(n)]
    for future in concurrent.futures.as_completed(futures):
        i, x = future.result()
        A_inv[:, i] = x

print("Approximate inverse of A:")
print(A_inv)
