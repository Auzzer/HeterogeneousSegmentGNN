# This version only contains structural edges (x, y, z directions) for a 3D volumetric cloth.
import argparse
import numpy as np
import taichi as ti

@ti.data_oriented
class Cube3D:
    def __init__(self, N=8):
        """
        Creates a 3D grid of size N=8 => (N+1)^3 = 9^3 = 729 vertices.
        Only structural edges are kept (x, y, z directions).
        """
        self.N = N
        self.NV = (N + 1) ** 3  # total number of vertices
        
        # Number of structural edges in a 3D grid:
        # For each of the 3 directions, we have N*(N+1)*(N+1)
        self.NE = 3 * N * (N + 1) * (N + 1)
        
        # 3D vector fields
        self.pos = ti.Vector.field(3, ti.f32, self.NV)
        self.initPos = ti.Vector.field(3, ti.f32, self.NV)
        self.vel = ti.Vector.field(3, ti.f32, self.NV)
        self.force = ti.Vector.field(3, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)

        # Flattened 1D arrays for the solver
        self.vel_1D = ti.ndarray(ti.f32, 3 * self.NV)
        self.force_1D = ti.ndarray(ti.f32, 3 * self.NV)
        self.b = ti.ndarray(ti.f32, 3 * self.NV, needs_grad=True)

        # Spring data
        self.spring = ti.Vector.field(2, ti.i32, self.NE)  # each edge: (v1, v2)
        self.indices = ti.field(ti.i32, 2 * self.NE)       # for line visualization
        self.Jx = ti.Matrix.field(3, 3, ti.f32, self.NE)   # 3×3 derivative wrt positions
        self.Jv = ti.Matrix.field(3, 3, ti.f32, self.NE)   # 3×3 derivative wrt velocities
        self.rest_len = ti.field(ti.f32, self.NE)
        self.spring_ks = ti.field(ti.f32, self.NE)         # stiffness per edge

        # Global parameters
        self.kd = 0.5     # damping
        self.kf = 1e7     # fix point stiffness
        self.gravity = ti.Vector([0.0, -9, 0.0])

        # Initialize data
        self.init_positions()
        self.init_edges()
        self.init_spring_stiffness()

        # Build mass matrix (3*NV x 3*NV)
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NV, max_num_triplets=100000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()

        # Builders for damping & stiffness
        self.DBuilder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NV, max_num_triplets=50000000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NV, max_num_triplets=50000000)

        # Fix the top layer j=N
        # Vertex ID for (i, j, k) is i*(N+1)^2 + j*(N+1) + k
        fix_list = []
        for i in range(N+1):
            for k in range(N+1):
                j = N  # top layer
                v_id = i*(N+1)*(N+1) + j*(N+1) + k
                fix_list.append(v_id)
        self.fix_vertex_list = fix_list
        self.Jf = ti.Matrix.field(3, 3, ti.f32, len(self.fix_vertex_list))
        self.num_fixed_vertices = len(self.fix_vertex_list)

    @ti.kernel
    def init_positions(self):
        """
        Place vertices in a 3D grid from (0.25, 0.75, 0.25) to about (0.75, 1.25, 0.75).
        You can adjust these ranges as you like.
        """
        for i, j, k in ti.ndrange(self.N+1, self.N+1, self.N+1):
            idx = i*(self.N+1)*(self.N+1) + j*(self.N+1) + k

            x_coord = 0.25 + 0.5 * (i / self.N)
            y_coord = 0.75 + 0.5 * (j / self.N)
            z_coord = 0.25 + 0.5 * (k / self.N)

            self.pos[idx] = ti.Vector([x_coord, y_coord, z_coord])
            self.initPos[idx] = self.pos[idx]
            self.vel[idx] = ti.Vector.zero(ti.f32, 3)
            self.mass[idx] = 0.2  # can be changed

    @ti.kernel
    def init_edges(self):
        """
        Only structural edges in 3D:
            (i, j, k) to (i+1, j, k)
            (i, j, k) to (i, j+1, k)
            (i, j, k) to (i, j, k+1)
        """
        e = 0
        for i, j, k in ti.ndrange(self.N+1, self.N+1, self.N+1):
            idx = i*(self.N+1)*(self.N+1) + j*(self.N+1) + k
            
            # x-direction edge
            if i < self.N:
                idx2 = (i+1)*(self.N+1)*(self.N+1) + j*(self.N+1) + k
                self.spring[e] = ti.Vector([idx, idx2])
                self.rest_len[e] = (self.pos[idx] - self.pos[idx2]).norm()
                e += 1
                
            # y-direction edge
            if j < self.N:
                idx2 = i*(self.N+1)*(self.N+1) + (j+1)*(self.N+1) + k
                self.spring[e] = ti.Vector([idx, idx2])
                self.rest_len[e] = (self.pos[idx] - self.pos[idx2]).norm()
                e += 1
                
            # z-direction edge
            if k < self.N:
                idx2 = i*(self.N+1)*(self.N+1) + j*(self.N+1) + (k+1)
                self.spring[e] = ti.Vector([idx, idx2])
                self.rest_len[e] = (self.pos[idx] - self.pos[idx2]).norm()
                e += 1

    @ti.kernel
    def init_spring_stiffness(self):
        # Example: randomize around [1000, 1500]
        for e in range(self.NE):
            self.spring_ks[e] = 1000.0 + 200.0 * ti.random()

    @ti.kernel
    def init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
        # Put mass on the diagonal
        for i in range(self.NV):
            m = self.mass[i]
            for c in ti.static(range(3)):
                M[3*i + c, 3*i + c] += m

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        # Gravity
        for i in range(self.NV):
            self.force[i] += self.gravity * self.mass[i]
        # Structural edges only
        for i in range(self.NE):
            idx1 = self.spring[i][0]
            idx2 = self.spring[i][1]
            p1 = self.pos[idx1]
            p2 = self.pos[idx2]
            dist = p2 - p1
            length = dist.norm()
            if length > 1e-12:
                dir = dist / length
                stretch = length - self.rest_len[i]
                f = self.spring_ks[i] * stretch * dir
                self.force[idx1] += f
                self.force[idx2] -= f

        # Fix top layer with penalty force
        for idx in ti.static(range(self.num_fixed_vertices)):
            v_id = self.fix_vertex_list[idx]
            self.force[v_id] += self.kf * (self.initPos[v_id] - self.pos[v_id])

    @ti.kernel
    def compute_Jacobians(self):
        # For each structural edge
        for i in range(self.NE):
            idx1 = self.spring[i][0]
            idx2 = self.spring[i][1]
            p1 = self.pos[idx1]
            p2 = self.pos[idx2]
            dx = p1 - p2
            l = dx.norm()
            I3 = ti.Matrix.identity(ti.f32, 3)
            inv_l = 1.0 / l if l > 1e-12 else 0.0
            # dx dx^T
            dxdxT = dx.outer_product(dx)
            # stiffness derivative
            # Jx = k * [I - (rest_len / l) * (I - (dx dx^T)/l^2 )]
            r0 = self.rest_len[i]
            factor = r0 * inv_l
            self.Jx[i] = self.spring_ks[i] * (I3 - factor * (I3 - dxdxT * (inv_l**2)))

            # velocity Jacobian = kd * I3
            self.Jv[i] = self.kd * I3

        # Fixed vertices: add -kf on the diagonal
        for idx in ti.static(range(self.num_fixed_vertices)):
            self.Jf[idx] = -self.kf * ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        # Structural edges
        for i in range(self.NE):
            idx1 = self.spring[i][0]
            idx2 = self.spring[i][1]
            J = self.Jx[i]
            for r, c in ti.static(ti.ndrange(3, 3)):
                # vertex1 wrt vertex1
                K[3*idx1 + r, 3*idx1 + c] -= J[r, c]
                # vertex1 wrt vertex2
                K[3*idx1 + r, 3*idx2 + c] += J[r, c]
                # vertex2 wrt vertex1
                K[3*idx2 + r, 3*idx1 + c] += J[r, c]
                # vertex2 wrt vertex2
                K[3*idx2 + r, 3*idx2 + c] -= J[r, c]

        # Fixed vertices
        for f_idx in ti.static(range(self.num_fixed_vertices)):
            v_id = self.fix_vertex_list[f_idx]
            Jf_mat = self.Jf[f_idx]
            for r, c in ti.static(ti.ndrange(3, 3)):
                K[3*v_id + r, 3*v_id + c] += Jf_mat[r, c]

    @ti.kernel
    def assemble_D(self, D: ti.types.sparse_matrix_builder()):
        # Damping from each edge
        for i in range(self.NE):
            idx1 = self.spring[i][0]
            idx2 = self.spring[i][1]
            Jd = self.Jv[i]
            for r, c in ti.static(ti.ndrange(3, 3)):
                D[3*idx1 + r, 3*idx1 + c] -= Jd[r, c]
                D[3*idx1 + r, 3*idx2 + c] += Jd[r, c]
                D[3*idx2 + r, 3*idx1 + c] += Jd[r, c]
                D[3*idx2 + r, 3*idx2 + c] -= Jd[r, c]

    @ti.kernel
    def updatePosVel(self, h: ti.f32, dv: ti.types.ndarray()):
        # dv is the increment in velocity
        for i in range(self.NV):
            vx = dv[3*i + 0]
            vy = dv[3*i + 1]
            vz = dv[3*i + 2]
            self.vel[i] += ti.Vector([vx, vy, vz])
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def copy_to(self, des: ti.types.ndarray(), source: ti.template()):
        # flatten 3D vector -> 3*N
        for i in range(self.NV):
            des[3*i + 0] = source[i][0]
            des[3*i + 1] = source[i][1]
            des[3*i + 2] = source[i][2]

    @ti.kernel
    def compute_b(self,
                  b: ti.types.ndarray(),
                  f: ti.types.ndarray(),
                  Kv: ti.types.ndarray(),
                  h: ti.f32):
        # b = h * (f + K*v*h)
        for i in range(3*self.NV):
            b[i] = h * (f[i] + Kv[i] * h)

    @ti.kernel
    def spring2indices(self):
        """
        For line rendering in 3D. We store each edge in pairs.
        """
        for i in range(self.NE):
            self.indices[2*i]   = self.spring[i][0]
            self.indices[2*i+1] = self.spring[i][1]

    def update(self, h):
        # 1. Compute forces
        self.compute_force()

        # 2. Compute Jacobians
        self.compute_Jacobians()

        # 3. Assemble D and K
        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()
        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()

        # 4. A = M - h*D - h^2*K
        A = self.M - (h * D) - (h**2) * K

        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)

        # b = h*(f + K*v*h)
        Kv = K @ self.vel_1D
        self.compute_b(self.b, self.force_1D, Kv, h)

        # 5. Solve
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        dv = solver.solve(self.b)

        # 6. Update x, v
        self.updatePosVel(h, dv)

    def display(self, gui, radius=3, color=0xFFFFFF):
        """
        2D GUI lines for debugging: project (x,y,z) -> (x, z) or (x,y).
        (This is purely for older 2D ti.GUI – you can skip or adapt for GGUI.)
        """
        lines = self.spring.to_numpy()
        pos_np = self.pos.to_numpy()
        edge_begin = np.zeros((lines.shape[0], 2), dtype=np.float32)
        edge_end   = np.zeros((lines.shape[0], 2), dtype=np.float32)
        for i in range(lines.shape[0]):
            idx1, idx2 = lines[i]
            x1, y1, z1 = pos_np[idx1]
            x2, y2, z2 = pos_np[idx2]
            # e.g., project to x-z
            edge_begin[i] = [x1, z1]
            edge_end[i]   = [x2, z2]

        gui.lines(edge_begin, edge_end, radius=1, color=0x0000FF)
        circles_pos = np.array([[p[0], p[2]] for p in pos_np])
        gui.circles(circles_pos, radius=radius, color=color)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--use-ggui", action="store_true", help="Use GGUI")
    parser.add_argument("-a", "--arch", default="cpu", type=str, help="Backend")
    args = parser.parse_args()

    if args.arch in ["cpu", "x64", "arm64"]:
        ti.init(arch=ti.cpu, random_seed=666)
    elif args.arch in ["cuda", "gpu"]:
        ti.init(arch=ti.cuda, random_seed=666)
    else:
        raise ValueError("Only CPU/CUDA supported in this snippet.")

    # Create a 3D volumetric cloth of size 8x8x8
    cube_3d = Cube3D()
    h = 0.03
    cube_3d.spring2indices()

    # GGUI display
    window = ti.ui.Window("3D Volumetric Cloth (Structural Only)", (800, 800))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))

    while window.running:
        cube_3d.update(h)

        # Set camera
        camera.position(1.5, 1.5, 1.5)
        camera.lookat(0.5, 1.0, 0.5)
        scene.set_camera(camera)

        # Draw edges (lines) and particles
        scene.lines(cube_3d.pos, indices=cube_3d.indices, color=(0, 0, 1), width=0.002)
        scene.particles(cube_3d.pos, radius=0.005, color=(0, 0, 1))

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()
