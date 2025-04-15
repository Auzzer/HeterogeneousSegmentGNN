import taichi as ti
arch = ti.cpu # or ti.cuda
ti.init(arch=arch)
n = 4
k = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
k[None] = 10  # Initialize value


#-————————————————————————————————————————————————————————————————————————————————
A_build = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=10)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    for i in range(n):
        A[i, i] += 0.2*i*k[None]

fill(A_build)
A1 = A_build.build()
print(A1)


#-————————————————————————————————————————————————————————————————————————————————
#A2
A2 = ti.field(dtype=ti.f32, shape=(n, n), needs_grad=True)

@ti.kernel
def fill_dense():
    for i in range(n):
            A2[i, i] = 0.2*i*k[None]

fill_dense()

print(A2)



# Define a scalar loss (sum of A2)
loss2 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_loss2():
    for i, j in A2:
        loss2[None] += A2[i, j]

# Forward pass and gradient computation
with ti.ad.Tape(loss=loss2):
    fill_dense()    
    compute_loss2()  # Compute loss as sum(A2)

# Gradient of loss w.r.t k is now in k.grad[None]
print("dA2/dk (summed):", k.grad[None])


#-————————————————————————————————————————————————————————————————————————————————

# A1

# Define a scalar loss (sum of A1)
loss1 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
print(A1[1, 1])
@ti.kernel
def compute_loss1():
    for i in range(n):
        loss1[None] += A1[i, i]

# Forward pass and gradient computation
with ti.ad.Tape(loss=loss1):
    fill(A_build)
    A1 = A_build.build()
    print(A1[1, 1])    
    compute_loss1()  # Compute loss as sum(A2)

# Gradient of loss w.r.t k is now in k.grad[None]
print("dA2/dk (summed):", k.grad[None])

