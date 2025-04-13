# https://www.cs.cmu.edu/~baraff/papers/sig98.pdf
import argparse

import numpy as np

import taichi as ti


@ti.data_oriented
class Cloth:
    def __init__(self, N):
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1) ** 2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # numbser of edges
        self.pos = ti.Vector.field(2, ti.f32, self.NV)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)
        self.vel_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.force_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.b = ti.ndarray(ti.f32, 2 * self.NV)

        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.indices = ti.field(ti.i32, 2 * self.NE)
        self.Jx = ti.Matrix.field(2, 2, ti.f32, self.NE)  # Jacobian with respect to position
        self.Jv = ti.Matrix.field(2, 2, ti.f32, self.NE)  # Jacobian with respect to velocity
        self.rest_len = ti.field(ti.f32, self.NE)
        self.ks = 10000.0  # spring stiffness
        self.spring_ks = ti.field(ti.f32, self.NE)
        self.kd = 0.5  # damping constant
        self.kf = 1.0e5  # fix point stiffness

        self.gravity = ti.Vector([0.0, -2.0])
        self.init_pos()
        self.init_edges()
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=10000000)
        self.DBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=10000000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=100000000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()
        self.fix_vertex_list = [i * (N + 1) + N for i in range(N + 1)]
        self.Jf = ti.Matrix.field(2, 2, ti.f32, len(self.fix_vertex_list))
        self.num_fixed_vertices = len(self.fix_vertex_list)


        

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.pos[k] = ti.Vector([i, j]) / self.N * 0.5 + ti.Vector([0.25, 0.25])
            self.initPos[k] = self.pos[k]
            self.vel[k] = ti.Vector([0, 0])
            self.mass[k] = 1.0

    @ti.kernel
    def init_edges(self):
        pos, spring, N, rest_len = ti.static(self.pos, self.spring, self.N, self.rest_len)
        for i, j in ti.ndrange(N + 1, N):
            idx, idx1 = i * N + j, i * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx1 + 1])
            rest_len[idx] = (pos[idx1] - pos[idx1 + 1]).norm()
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            idx, idx1, idx2 = (
                start + i + j * N,
                i * (N + 1) + j,
                i * (N + 1) + j + N + 1,
            )
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            idx, idx1, idx2 = (
                start + i * N + j,
                i * (N + 1) + j,
                (i + 1) * (N + 1) + j + 1,
            )
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
        start = 2 * N * (N + 1) + N * N
        for i, j in ti.ndrange(N, N):
            idx, idx1, idx2 = (
                start + i * N + j,
                i * (N + 1) + j + 1,
                (i + 1) * (N + 1) + j,
            )
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

    @ti.kernel
    def init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
        for i in range(self.NV):
            mass = self.mass[i]
            M[2 * i + 0, 2 * i + 0] += mass
            M[2 * i + 1, 2 * i + 1] += mass

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        for i in self.force:
            self.force[i] += self.gravity * self.mass[i]

        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dis = pos2 - pos1
            force = self.ks * (dis.norm() - self.rest_len[i]) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force
        # fix constraint gradient
        for idx in ti.static(self.fix_vertex_list):
            self.force[idx] += self.kf * (self.initPos[idx] - self.pos[idx])


    @ti.kernel
    def compute_Jacobians(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1]], [dx[1] * dx[0], dx[1] * dx[1]]])
            l = dx.norm()
            if l != 0.0:
                l = 1.0 / l
            self.Jx[i] = (I - self.rest_len[i] * l * (I - dxtdx * l**2)) * self.ks
            self.Jv[i] = self.kd * I

        # fix point constraint hessian
        for idx in ti.static(range(self.num_fixed_vertices)):
            self.Jf[idx] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])


    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * idx1 + m, 2 * idx1 + n] -= self.Jx[i][m, n]
                K[2 * idx1 + m, 2 * idx2 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx1 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx2 + n] -= self.Jx[i][m, n]
        for idx in ti.static(range(self.num_fixed_vertices)):
            vertex_idx = self.fix_vertex_list[idx]
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * vertex_idx + m, 2 * vertex_idx + n] += self.Jf[idx][m, n]


    @ti.kernel
    def assemble_D(self, D: ti.types.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                D[2 * idx1 + m, 2 * idx1 + n] -= self.Jv[i][m, n]
                D[2 * idx1 + m, 2 * idx2 + n] += self.Jv[i][m, n]
                D[2 * idx2 + m, 2 * idx1 + n] += self.Jv[i][m, n]
                D[2 * idx2 + m, 2 * idx2 + n] -= self.Jv[i][m, n]

    @ti.kernel
    def updatePosVel(self, h: ti.f32, dv: ti.types.ndarray()):
        for i in self.pos:
            self.vel[i] += ti.Vector([dv[2 * i], dv[2 * i + 1]])
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def copy_to(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.NV):
            des[2 * i] = source[i][0]
            des[2 * i + 1] = source[i][1]

    @ti.kernel
    def compute_b(
        self,
        b: ti.types.ndarray(),
        f: ti.types.ndarray(),
        Kv: ti.types.ndarray(),
        h: ti.f32,
    ):
        for i in range(2 * self.NV):
            b[i] = (f[i] + Kv[i] * h) * h

    def update(self, h):
        self.compute_force()

        self.compute_Jacobians()
        # Assemble global system

        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()

        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()

        A = self.M - h * D - h**2 * K

        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)

        # b = (force + h * K @ vel) * h
        Kv = K @ self.vel_1D
        self.compute_b(self.b, self.force_1D, Kv, h)

        # Sparse solver
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        # Solve the linear system
        dv = solver.solve(self.b)
        self.updatePosVel(h, dv)
        
    
    def display(self, gui, radius=2, color=0xFFFFFF):
        lines = self.spring.to_numpy()
        pos = self.pos.to_numpy()
        edgeBegin = np.zeros(shape=(lines.shape[0], 2))
        edgeEnd = np.zeros(shape=(lines.shape[0], 2))
        for i in range(lines.shape[0]):
            idx1, idx2 = lines[i][0], lines[i][1]
            edgeBegin[i] = pos[idx1]
            edgeEnd[i] = pos[idx2]
        gui.lines(edgeBegin, edgeEnd, radius=2, color=0x0000FF)
        gui.circles(self.pos.to_numpy(), radius, color)
    """
    def display(self, gui, radius=5, color=0xFFFFFF):
    
    
        # 1. 拿到布料的所有节点坐标（注意：pos 是 taichi.field，需要转到 numpy）
        pos_np = self.pos.to_numpy()  # shape = [N, 2]

        # 2. 计算包围盒 (x_min, x_max, y_min, y_max)
        x_min, y_min = pos_np.min(axis=0)
        x_max, y_max = pos_np.max(axis=0)

        # 避免出现除 0 的情况
        eps = 1e-5
        dx = max(x_max - x_min, eps)
        dy = max(y_max - y_min, eps)

        # 3. 做一个归一化：将 (x_min, x_max)→(0,1)，(y_min, y_max)→(0,1)
        #   （如果你不想拉伸变形，可以用同一个 scale = 1 / max(dx, dy)，然后在另一维留边）
        pos_gui = np.zeros_like(pos_np)
        pos_gui[:, 0] = (pos_np[:, 0] - x_min) / dx
        pos_gui[:, 1] = (pos_np[:, 1] - y_min) / dy

        # 也可多预留一点边距：
        margin = 0.05
        pos_gui = pos_gui * (1 - 2 * margin) + margin

        # 4. 同理把每条弹簧线的两个端点也映射到 [0,1]×[0,1]
        lines = self.spring.to_numpy()  # shape = [M, 2], 每行是 (idx1, idx2)
        edge_begin = np.zeros((lines.shape[0], 2))
        edge_end   = np.zeros((lines.shape[0], 2))

        for i in range(lines.shape[0]):
            idx1, idx2 = lines[i]
            edge_begin[i] = pos_gui[idx1]
            edge_end[i]   = pos_gui[idx2]

        # 5. 绘制线和点（已经在 [0,1] 范围内）
        gui.lines(edge_begin, edge_end, radius=2, color=0x0000FF)
        gui.circles(pos_gui, radius, color)
    """

    @ti.kernel
    def spring2indices(self):
        for i in self.spring:
            self.indices[2 * i + 0] = self.spring[i][0]
            self.indices[2 * i + 1] = self.spring[i][1]

    def displayGGUI(self, canvas, radius=0.01, color=(1.0, 1.0, 1.0)):
        self.spring2indices()
        canvas.lines(self.pos, width=0.005, indices=self.indices, color=(0.0, 0.0, 1.0))
        canvas.circles(self.pos, radius, color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--use-ggui", action="store_true", help="Display with GGUI")
    parser.add_argument(
        "-a",
        "--arch",
        required=False,
        default="cuda",
        dest="arch",
        type=str,
        help="The arch (backend) to run this example on",
    )
    args, unknowns = parser.parse_known_args()
    arch = args.arch
    if arch in ["x64", "cpu", "arm64"]:
        ti.init(arch=ti.cpu)
    elif arch in ["cuda", "gpu"]:
        ti.init(arch=ti.cuda)
    else:
        raise ValueError("Only CPU and CUDA backends are supported for now.")

    h = 0.01
    pause = False
    cloth = Cloth(N=64)

    use_ggui = args.use_ggui
    if not use_ggui:
        gui = ti.GUI("Implicit Mass Spring System", res=(500, 500))
        while gui.running:
            for e in gui.get_events():
                if e.key == gui.ESCAPE:
                    gui.running = False
                elif e.key == gui.SPACE:
                    pause = not pause

            if not pause:
                cloth.update(h)

            cloth.display(gui)
            gui.show()
    else:
        window = ti.ui.Window("Implicit Mass Spring System", res=(500, 500))
        while window.running:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.ESCAPE:
                    break
            if window.is_pressed(ti.ui.SPACE):
                pause = not pause

            if not pause:
                cloth.update(h)

            canvas = window.get_canvas()
            cloth.displayGGUI(canvas)
            window.show()


if __name__ == "__main__":
    main()