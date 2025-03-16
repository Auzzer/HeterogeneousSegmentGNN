import argparse
import numpy as np
import taichi as ti

@ti.data_oriented
class Cloth3D:
    def __init__(self, N):
        self.N = N
        
        # total number of vertices = (N+1)*(N+1), but each vertex is now in 3D
        self.NV = (N + 1) ** 2  
        
        # total number of edges (structural + shear) is the same count as 2D:
        self.NE = 2 * N * (N + 1) + 2 * N * N  # structural + shear
        self.NB = 2 * (N + 1) * (N - 1)        # bending edges
        self.NE_total = self.NE + self.NB
        
        # 3D vector fields
        self.pos = ti.Vector.field(3, ti.f32, self.NV)
        self.initPos = ti.Vector.field(3, ti.f32, self.NV)
        self.vel = ti.Vector.field(3, ti.f32, self.NV)
        self.force = ti.Vector.field(3, ti.f32, self.NV)
        
        # mass is still a scalar
        self.mass = ti.field(ti.f32, self.NV)
        
        # Flattened 1D arrays (3D -> 3 * NV) for the solver
        self.vel_1D = ti.ndarray(ti.f32, 3 * self.NV)
        self.force_1D = ti.ndarray(ti.f32, 3 * self.NV)
        self.b = ti.ndarray(ti.f32, 3 * self.NV, needs_grad=True)

        # Spring arrays
        self.spring = ti.Vector.field(2, ti.i32, self.NE_total)
        self.indices = ti.field(ti.i32, 2 * self.NE)  # for line visualization
        self.Jx = ti.Matrix.field(3, 3, ti.f32, self.NE_total)  # 3x3 per spring (pos derivative)
        self.Jv = ti.Matrix.field(3, 3, ti.f32, self.NE_total)  # 3x3 per spring (vel derivative)
        self.rest_len = ti.field(ti.f32, self.NE_total)
        
        # Per-edge stiffness
        self.spring_ks = ti.field(ti.f32, self.NE_total)
        # self.bend_ks = ti.field(ti.f32, self.NB)

        # Global parameters
        self.kd = 0.5     # damping
        self.kf = 1e7     # fix point stiffness
        self.gravity = ti.Vector([0.0, -2, 0.0])  # 3D gravity

        # Initialize
        self.init_positions()
        self.init_edges()
        self.init_spring_stiffness()
        self.init_bend_stiffness()
        
        # Build mass matrix (3*NV x 3*NV)
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NV, max_num_triplets=1000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()

        # Builders for damping & stiffness
        self.DBuilder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NV, max_num_triplets=100000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NV, max_num_triplets=100000)

        # Let’s fix the top row of the grid to maintain shape
        # e.g., all vertices with i=0 or i=N (you can choose whichever you prefer).
        # For demonstration, let's fix row i=0. That means j=0..N for i=0
        self.fix_vertex_list = [i * (N+1) + N for i in range(N+1)]
        self.Jf = ti.Matrix.field(3, 3, ti.f32, len(self.fix_vertex_list))
        self.num_fixed_vertices = len(self.fix_vertex_list)

    @ti.kernel
    def init_positions(self):
        """
        Initialize the cloth in the X-Z plane (y=0) or whichever plane you prefer.
        For example, place the cloth from (0,0,0) to (0,0,0.5) in z, and from 0..0.5 in x.
        """
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            x_coord = i / self.N * 0.5 + 0.25
            # Let the y-coordinate vary so that the top row (j = N) is higher.
            y_coord = j / self.N * 0.5 + 0.75  # top row at y=1.25, bottom row at y=0.75
            z_coord = 0.25
            self.pos[k] = ti.Vector([x_coord, y_coord, z_coord])
            self.initPos[k] = self.pos[k]
            self.vel[k] = ti.Vector([0.0, 0.0, 0.0])
            self.mass[k] = 0.2  # or random if desired

    @ti.kernel
    def init_edges(self):
        """
        Same indexing logic as the 2D cloth, but the pos is 3D now.
        Structural: (i,j)-(i,j+1), (i,j)-(i+1,j)
        Shearing: (i,j)-(i+1,j+1), (i+1,j)-(i,j+1)
        Bending: (i,j)-(i,j+2), (i,j)-(i+2,j)
        """
        pos, spring, N, rest_len = ti.static(self.pos, self.spring, self.N, self.rest_len)

        # --- structural horizontal ---
        for i, j in ti.ndrange(N+1, N):
            idx = i*N + j
            idx1 = i*(N+1) + j
            idx2 = i*(N+1) + (j+1)
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

        # --- structural vertical ---
        start_v = N*(N+1)
        for i, j in ti.ndrange(N, N+1):
            idx = start_v + i + j*N
            idx1 = i*(N+1) + j
            idx2 = (i+1)*(N+1) + j
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

        # --- shear: (i,j)-(i+1,j+1) ---
        start_shear_1 = 2*N*(N+1)
        for i, j in ti.ndrange(N, N):
            idx = start_shear_1 + i*N + j
            idx1 = i*(N+1) + j
            idx2 = (i+1)*(N+1) + (j+1)
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

        # --- shear: (i+1,j)-(i,j+1) ---
        start_shear_2 = 2*N*(N+1) + N*N
        for i, j in ti.ndrange(N, N):
            idx = start_shear_2 + i*N + j
            idx1 = (i+1)*(N+1) + j
            idx2 = i*(N+1) + (j+1)
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

        # Bending
        bend_start = 2*N*(N+1) + 2*N*N
        # horizontal bending: (i,j)-(i,j+2)
        for i, j in ti.ndrange(N+1, N-1):
            idx = bend_start + i*(N-1) + j
            idx1 = i*(N+1) + j
            idx2 = i*(N+1) + (j+2)
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

        # vertical bending: (i,j)-(i+2,j)
        horiz_bend_count = (N+1)*(N-1)
        for i, j in ti.ndrange(N-1, N+1):
            idx = bend_start + horiz_bend_count + i*(N+1) + j
            idx1 = i*(N+1) + j
            idx2 = (i+2)*(N+1) + j
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

    @ti.kernel
    def init_spring_stiffness(self):
        # Example: randomize within [1000, 1300]
        for e in range(self.NE):
            self.spring_ks[e] = 1000.0 + 500.0 * ti.random()

    @ti.kernel
    def init_bend_stiffness(self):
        # set bend stiffness to ~0.1 * average structural
        avg = 0.0
        for e in range(self.NE):
            avg += self.spring_ks[e]
        avg /= self.NE
        for i in range(self.NB):
            self.spring_ks[i+self.NE] = 0.1 * avg

    @ti.kernel
    def init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
        for i in range(self.NV):
            m = self.mass[i]
            # place mass on the diagonal 3x3 block
            for c in ti.static(range(3)):
                M[3*i + c, 3*i + c] += m

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        # Add gravity
        for i in range(self.NV):
            self.force[i] += self.gravity * self.mass[i]

        # Structural + Shear
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

        # Bending
        for i in range(self.NE, self.NE_total):
            idx1 = self.spring[i][0]
            idx2 = self.spring[i][1]
            p1 = self.pos[idx1]
            p2 = self.pos[idx2]
            dist = p2 - p1
            length = dist.norm()
            if length > 1e-12:
                dir = dist / length
                stretch = length - self.rest_len[i]
                # get the bending stiffness index
                f = self.spring_ks[i] * stretch * dir
                self.force[idx1] += f
                self.force[idx2] -= f

        # Fix top row constraints with a simple penalty force:
        for idx in ti.static(range(self.num_fixed_vertices)):
            v_id = self.fix_vertex_list[idx]
            # penalty force = kf * (initial_position - current_position)
            self.force[v_id] += self.kf * (self.initPos[v_id] - self.pos[v_id])

    @ti.kernel
    def compute_Jacobians(self):
        # Positional Jacobians
        for i in range(self.NE_total):
            idx1 = self.spring[i][0]
            idx2 = self.spring[i][1]
            p1 = self.pos[idx1]
            p2 = self.pos[idx2]
            dx = p1 - p2
            l = dx.norm()
            I3 = ti.Matrix.identity(ti.f32, 3)
            if l > 1e-12:
                l = 1.0 / l
            else:
                l = 0.0
            

            # same formula extended to 3D:
            # Jx = k_curr * ( I - (rest_len/l) * ( I - (dx dx^T)/l^2 ) )
            # but carefully: rest_len[i] vs l
            r0 = self.rest_len[i]
            dxdxT = dx.outer_product(dx)  # 3x3
            # 修正后的公式
            self.Jx[i] = (I3-self.rest_len[i]*l* (I3 - dxdxT*l**2))*self.spring_ks[i]

            # Velocity Jacobian = kd * I3 for each end, with sign pattern
            self.Jv[i] = self.kd * I3
        
        # The Jacobians for the fixed vertices:
        # effectively a block +kf * (-I3) on the diagonal
        for idx in ti.static(range(self.num_fixed_vertices)):
            self.Jf[idx] = -self.kf * ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        # Springs
        for i in range(self.NE_total):
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
                # add Jf to the diagonal of the fixed vertex
                K[3*v_id + r, 3*v_id + c] += Jf_mat[r, c]

    @ti.kernel
    def assemble_D(self, D: ti.types.sparse_matrix_builder()):
        # damping from each spring
        for i in range(self.NE_total):
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
        for i in range(3*self.NV):
            b[i] = (f[i] + Kv[i]*h)*h

    @ti.kernel
    def spring2indices(self):
        """
        For visualization lines: only the structural+shear edges (self.NE).
        Bending edges often overlap or create duplicates in visuals, so up to you.
        """
        for i in range(self.NE):
            self.indices[2*i]   = self.spring[i][0]
            self.indices[2*i+1] = self.spring[i][1]

    def update(self, h):
        # 1. Forces
        self.compute_force()

        # 2. Jacobians
        self.compute_Jacobians()

        # 3. Assemble damping & stiffness
        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()
        self.assemble_K(self.KBuilder)
        
        K = self.KBuilder.build()

        # 4. A = M - h*D - h^2*K
        A = self.M - (h * D) - (h**2)*K

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

        # 6. Update x,v
        self.updatePosVel(h, dv)

    def display(self, gui, radius=3, color=0xFFFFFF):
        """
        2D GUI lines: we can simply ignore one dimension or do some projection
        for quick debugging.  For example, we can show x-z if you prefer.
        """
        lines = self.spring.to_numpy()[:self.NE]  # structural + shear edges
        pos_np = self.pos.to_numpy()
        edge_begin = np.zeros((lines.shape[0], 2), dtype=np.float32)
        edge_end   = np.zeros((lines.shape[0], 2), dtype=np.float32)
        for i in range(lines.shape[0]):
            idx1, idx2 = lines[i]
            # project (x, y, z) -> (x, z) or (x, y), etc.
            x1, y1, z1 = pos_np[idx1]
            x2, y2, z2 = pos_np[idx2]
            edge_begin[i] = [x1, z1]  # e.g. show x-z plane
            edge_end[i]   = [x2, z2]

        gui.lines(edge_begin, edge_end, radius=1, color=0x0000FF)

        # Draw circles for vertices
        circles_pos = np.array([[p[0], p[2]] for p in pos_np])  # again x-z
        gui.circles(circles_pos, radius=radius, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--use-ggui", action="store_true", help="Use GGUI")
    parser.add_argument("-a", "--arch", default="gpu", type=str, help="Backend")
    args = parser.parse_args()

    if args.arch in ["cpu", "x64", "arm64"]:
        ti.init(arch=ti.cpu)
    elif args.arch in ["cuda", "gpu"]:
        ti.init(arch=ti.cuda)
    else:
        raise ValueError("Only CPU/CUDA supported in this snippet.")

    cloth_3d = Cloth3D(N=16)
    h = 0.01
    

    
    # GGUI display as points/lines in 3D
    window = ti.ui.Window("3D Cloth Simulation", (800, 800))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    cloth_3d.spring2indices()

    while window.running:
        cloth_3d.update(h)

        camera.position(0.5, 1.0, 1.5)
        camera.lookat(0.5, 1.0, 0.5)

        # center of cloth
        scene.set_camera(camera)

        # draw lines
        # 在 GGUI 中正确渲染 3D
        scene.lines(cloth_3d.pos, indices=cloth_3d.indices, color=(0, 0, 1), width=0.01)
        scene.particles(cloth_3d.pos, radius=0.005, color=(0, 0, 1))

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()
