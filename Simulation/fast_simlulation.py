import taichi as ti
ti.init(arch=ti.gpu)

# ---------------- 常量与参数 ----------------
n = 32  # 网格分辨率 (n x n)
dt = 1e-2  # 时间步长（隐式方法允许较大值）
gravity = ti.Vector([0.0, -9.8, 0.0])
stiffness = 1e4  # 弹簧刚度
damping = 0.99  # 全局阻尼系数

# 质点位置、速度、质量
x = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
v = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
mass = 1.0

# 弹簧连接信息（右和上）
spring_offsets = [ti.Vector([1, 0]), ti.Vector([0, 1])]

# 稀疏矩阵数据结构（用于隐式求解）
row = ti.field(ti.int32)
col = ti.field(ti.int32)
val = ti.field(ti.f32)
ti.root.dynamic(ti.i, n * n * 4).place(row, col, val)

# ---------------- 初始化 ----------------
@ti.kernel
def initialize():
    for i, j in x:
        x[i, j] = ti.Vector([i / n, 1.0, j / n])  # 初始位置在y=1.0平面
        v[i, j] = ti.Vector([0.0, 0.0, 0.0])

# ---------------- 构建刚度矩阵 K ----------------
@ti.kernel
def build_K():
    # 清空矩阵
    row.deactivate()
    col.deactivate()
    val.deactivate()
    
    for i, j in x:
        for offset in ti.static(spring_offsets):
            ni, nj = i + offset[0], j + offset[1]
            if 0 <= ni < n and 0 <= nj < n:
                # 弹簧两端的质点索引
                idx = i * n + j
                n_idx = ni * n + nj
                
                # 计算相对位移和力
                dx = x[i, j] - x[ni, nj]
                length = dx.norm()
                
                force = stiffness * (length - 1.0) * dx.normalized()
                
                # 雅可比矩阵 df/dx（隐式方法需要导数）
                # 这里简化为标量刚度贡献
                K_entry = stiffness * (dx.outer_product(dx) / (length ** 2))
                
                # 添加到全局矩阵（对称填充）
                for dim in ti.static(range(3)):
                    row.append(idx * 3 + dim)
                    col.append(n_idx * 3 + dim)
                    val.append(-K_entry[dim, dim])
                    row.append(idx * 3 + dim)
                    col.append(idx * 3 + dim)
                    val.append(K_entry[dim, dim])

# ---------------- 隐式积分求解 ----------------

def implicit_step():
    # 构建线性系统：(M - dt^2*K) * dv = dt*F + dt^2*gravity
    build_K()
    
    # 创建稀疏矩阵
    A = ti.linalg.SparseMatrix(n * n * 3, n * n * 3, row, col, val)
    
    # 右端项：dt*(gravity + external_forces)
    b = ti.ndarray(ti.f32, n * n * 3)
    for i, j in x:
        idx = i * n + j
        for d in ti.static(range(3)):
            b[idx * 3 + d] = dt * (mass * gravity[d] + ...)  # 此处需填充外力
    
    # 预条件共轭梯度法求解 A*dv = b
    dv = ti.ndarray(ti.f32, n * n * 3)
    ti.linalg.cg(A, b, dv, max_iter=100, tol=1e-6)
    
    # 更新速度和位置
    for i, j in x:
        idx = i * n + j
        for d in ti.static(range(3)):
            v[i, j][d] += dv[idx * 3 + d]
        v[i, j] *= damping  # 应用阻尼
        x[i, j] += v[i, j] * dt

# ---------------- 渲染 ----------------
window = ti.ui.Window("Implicit Mass-Spring", (800, 800))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

@ti.kernel
def update_vertices():
    for i, j in x:
        pos = x[i, j]
        canvas.set_pixel(i, j, ti.rgb_to_hex([pos.x, pos.y, pos.z]))

# ---------------- 主循环 ----------------
initialize()
while window.running:
    implicit_step()
    update_vertices()
    window.show()