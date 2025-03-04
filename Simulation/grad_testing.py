import taichi as ti

arch = ti.cpu # or ti.cuda
ti.init(arch=arch)

n = 4

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
A = ti.field(ti.f32, shape=(n, n), needs_grad=True)
for i in range(n):
    A[i, i] += 2.0
b = ti.field(ti.f32, shape=(n, ), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
x = ti.field(ti.f32, shape=(), needs_grad=True)
print(x.grad)
@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), b: ti.template(), interval: ti.i32):
    for i in range(n):
        A[i, i] += 2.0

        if i % interval == 0:
            b[i] += 1.0

# fill(K, b, 3)

# A = K.build()

N = 16

x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss2 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def func():
    temp = A@b
    for i in range(n):
        loss[None] += temp[i] **2
    



# Set the `grad` of the output variables to `1` before calling `func.grad()`.
loss.grad[None] = 1
loss2.grad[None] = 1

func()
func.grad()
