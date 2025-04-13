# Adding shearing and bending
# There are three types of edges: 1. structural, 2. shearing, 3. bending
# Structural: [i, j]-[i, j+1]; [i, j]-[i+1, j]} 
# Shear:[i j]-[i+1, j+1]; [i+1, j]-[i, j+1]} 
# Flexion (bend)[i, j]-[i, j+2] ;[i, j]-[i+2, j] 
# A visualization is: https://ics.uci.edu/~shz/courses/cs114/docs/proj3/index.html

import argparse
import numpy as np
import taichi as ti


@ti.data_oriented
class Cloth:
    def __init__(self, N):
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1) ** 2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # number of edges

        # -- Vertex data
        self.pos = ti.Vector.field(2, ti.f32, self.NV)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)

        # For sparse solver usage
        self.vel_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.force_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.b = ti.ndarray(ti.f32, 2 * self.NV)

        # -- Spring data
        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.indices = ti.field(ti.i32, 2 * self.NE)
        self.Jx = ti.Matrix.field(2, 2, ti.f32, self.NE)  # Jacobian w.r.t. position
        self.Jv = ti.Matrix.field(2, 2, ti.f32, self.NE)  # Jacobian w.r.t. velocity
        self.rest_len = ti.field(ti.f32, self.NE)

        # Heterogeneous spring stiffness: each edge can have its own stiffness
        self.spring_ks = ti.field(ti.f32, self.NE)   # <--- Heterogeneous stiffness

        # Global parameters (can remain global or partially heterogeneous if desired)
        self.kd = 0.5   # damping constant
        self.kf = 1.0e5 # fix point stiffness
        self.gravity = ti.Vector([0.0, -2.0])

        self.init_pos()
        self.init_edges()
        self.init_spring_stiffness()  # <--- Initialize per-edge stiffness

        # Build the mass matrix (sparse)
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=100000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()

        # Additional structures for damping & stiffness in global system
        self.DBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=100000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=100000)

        # Some of the vertices are fixed
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
            # Make the mass heterogeneous if desired:
            # E.g. random in [0.5, 1.5]
            self.mass[k] = 1.0 + 0.0 * ti.random()  # or 1.0 + 0.5 * ti.random()

    @ti.kernel
    def init_edges(self):
        pos, spring, N, rest_len = ti.static(self.pos, self.spring, self.N, self.rest_len)
        # Horizontal edges
        for i, j in ti.ndrange(N + 1, N):
            idx, idx1 = i * N + j, i * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx1 + 1])
            rest_len[idx] = (pos[idx1] - pos[idx1 + 1]).norm()

        # Vertical edges
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            idx, idx1, idx2 = (start + i + j * N,
                               i * (N + 1) + j,
                               i * (N + 1) + j + N + 1)
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

        # Diagonal edges (two directions)
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
        
        # Bending Springs
        # Compute the offset for bending springs.
        # (Total springs so far = structural + shear)
        bend_offset = 2 * N * (N + 1) + 2 * N * N

        # Horizontal bending springs: connect (i, j) to (i, j+2)
        # Valid j: 0 to N - 1 (since (N+1)-2 = N - 1), i from 0 to N
        for i, j in ti.ndrange(N + 1, N - 1):
            idx = bend_offset + i * (N - 1) + j
            idx1 = i * (N + 1) + j
            idx2 = i * (N + 1) + (j + 2)
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
            
        # Vertical bending springs: connect (i, j) to (i+2, j)
        # Valid i: 0 to N - 1 (since (N+1)-2 = N - 1), j from 0 to N+1
        # Compute the offset for vertical bending springs:
        horizontal_bend_count = (N + 1) * (N - 1)
        for i, j in ti.ndrange(N - 1, N + 1):
            idx = bend_offset + horizontal_bend_count + i * (N + 1) + j
            idx1 = i * (N + 1) + j
            idx2 = (i + 2) * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

    @ti.kernel
    def init_spring_stiffness(self):
        """
        Initialize a per-edge stiffness, making the cloth heterogeneous.
        You can make this random or follow any spatial pattern you like.
        """
        for e in range(self.NE):
            # E.g. randomly vary stiffness between [3e3, 6e3]:
            self.spring_ks[e] = 1500.0 + 300.0 * ti.random()

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
        # Gravity
        for i in self.force:
            self.force[i] += self.gravity * self.mass[i]

        # Spring forces (using heterogeneous spring_ks)
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dis = pos2 - pos1
            length = dis.norm()
            if length > 1e-12:  # avoid dividing by zero
                direction = dis / length
            else:
                direction = ti.Vector([0.0, 0.0])
            # Use per-edge stiffness:
            force_val = self.spring_ks[i] * (length - self.rest_len[i]) * dis.normalized()
            self.force[idx1] += force_val
            self.force[idx2] -= force_val

        # Fix point constraint gradient
        for idx in ti.static(self.fix_vertex_list):
            self.force[idx] += self.kf * (self.initPos[idx] - self.pos[idx])

    @ti.kernel
    def compute_Jacobians(self):
        # Spring Jacobians
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            dxtdx = ti.Matrix([
                [dx[0] * dx[0], dx[0] * dx[1]],
                [dx[1] * dx[0], dx[1] * dx[1]],
            ])
            l = dx.norm()
            inv_l = 0.0
            if l > 1e-12:
                inv_l = 1.0 / l

            # Incorporate per-edge stiffness spring_ks[i] instead of global ks:
            self.Jx[i] = (I - self.rest_len[i] * inv_l * (I - dxtdx * (inv_l**2))) * self.spring_ks[i]
            self.Jv[i] = self.kd * I

        # Fix point constraint Hessian
        for idx in ti.static(range(self.num_fixed_vertices)):
            self.Jf[idx] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])

    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        # Springs
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * idx1 + m, 2 * idx1 + n] -= self.Jx[i][m, n]
                K[2 * idx1 + m, 2 * idx2 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx1 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx2 + n] -= self.Jx[i][m, n]

        # Fixed points
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
        # 1. Compute force
        self.compute_force()
        # 2. Compute Jacobians
        self.compute_Jacobians()
        # 3. Assemble system
        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()

        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()

        # A = M - h*D - h^2*K
        A = self.M - h * D - (h**2) * K

        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)

        # b = (force + h*K*vel)*h
        Kv = K @ self.vel_1D
        self.compute_b(self.b, self.force_1D, Kv, h)

        # 4. Solve the linear system
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        dv = solver.solve(self.b)

        # 5. Update position & velocity
        self.updatePosVel(h, dv)

    @ti.kernel
    def spring2indices(self):
        for i in self.spring:
            self.indices[2 * i + 0] = self.spring[i][0]
            self.indices[2 * i + 1] = self.spring[i][1]

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
        default="cpu",
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
    cloth = Cloth(N=20)
    print(cloth.spring_ks.to_numpy())
    use_ggui = args.use_ggui
    if not use_ggui:
        gui = ti.GUI("Implicit Mass Spring System (Heterogeneous)", res=(500, 500))
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
        window = ti.ui.Window("Implicit Mass Spring System (Heterogeneous)", res=(500, 500))
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
