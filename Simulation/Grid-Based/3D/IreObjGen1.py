import argparse
import numpy as np
import taichi as ti

def generate_layered_shape_shrink(N=8, p_shrink=0.3):
    """
    Generate a layered shape on an (N+1)x(N+1)x(N+1) grid with:
      1. A randomly picked starting layer j0 (1 <= j0 < N) that is forced to be an 8×8 block.
      2. Upward and downward layers where the active region shrinks randomly.
    Returns:
      active_3d: boolean array with shape (N+1, N+1, N+1); True means the vertex is active.
    """
    grid_size = N + 1
    active_3d = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    j0 = np.random.randint(1, N)
    print(f"Chosen starting layer: {j0}")
    
    # For N=8, define an 8×8 block on layer j0 (i and k indices 1..8)
    i_min0, i_max0 = 1, grid_size
    k_min0, k_max0 = 1, grid_size
    for i in range(i_min0, i_max0):
        for k in range(k_min0, k_max0):
            active_3d[i, j0, k] = True

    def shrink_bound(i_min, i_max, k_min, k_max):
        if np.random.rand() < p_shrink:
            i_min += 1
        if np.random.rand() < p_shrink:
            i_max -= 1
        if np.random.rand() < p_shrink:
            k_min += 1
        if np.random.rand() < p_shrink:
            k_max -= 1
        if (i_max - i_min) < 2:
            i_min, i_max = i_min0, i_max0
        if (k_max - k_min) < 2:
            k_min, k_max = k_min0, k_max0
        return i_min, i_max, k_min, k_max

    # Upward layers: j0+1..N
    curr_i_min, curr_i_max = i_min0, i_max0
    curr_k_min, curr_k_max = k_min0, k_max0
    for j in range(j0+1, grid_size):
        curr_i_min, curr_i_max, curr_k_min, curr_k_max = shrink_bound(curr_i_min, curr_i_max, curr_k_min, curr_k_max)
        for i in range(curr_i_min, curr_i_max):
            for k in range(curr_k_min, curr_k_max):
                active_3d[i, j, k] = True

    # Downward layers: j0-1..0
    curr_i_min, curr_i_max = i_min0, i_max0
    curr_k_min, curr_k_max = k_min0, k_max0
    for j in range(j0-1, -1, -1):
        curr_i_min, curr_i_max, curr_k_min, curr_k_max = shrink_bound(curr_i_min, curr_i_max, curr_k_min, curr_k_max)
        for i in range(curr_i_min, curr_i_max):
            for k in range(curr_k_min, curr_k_max):
                active_3d[i, j, k] = True

    return active_3d


@ti.data_oriented
class Cube3D:
    def __init__(self, N, active_3d):
        """
        Build a 3D structural system (with only structural edges) but only for the vertices
        that are active according to active_3d.
          1) Reindex active vertices.
          2) Build the active positions and mass.
          3) Build structural edges among active neighbors.
          4) Pin the top layer (j=N) if the vertex is active.
        """
        self.N = N
        self._build_active_vertices(active_3d)
        self._build_edges_structural(active_3d)

        # Allocate Taichi fields for the active system.
        self.n_verts = self.pos_np.shape[0]   # number of active vertices
        self.n_edges = self.edges_np.shape[0]

        self.pos = ti.Vector.field(3, ti.f32, self.n_verts)
        self.initPos = ti.Vector.field(3, ti.f32, self.n_verts)
        self.vel = ti.Vector.field(3, ti.f32, self.n_verts)
        self.force = ti.Vector.field(3, ti.f32, self.n_verts)
        self.mass = ti.field(ti.f32, self.n_verts)

        # Flattened arrays for the solver.
        self.vel_1D = ti.ndarray(ti.f32, 3 * self.n_verts)
        self.force_1D = ti.ndarray(ti.f32, 3 * self.n_verts)
        self.b = ti.ndarray(ti.f32, 3 * self.n_verts, needs_grad=True)
        # Delta correction for collision
        self.delta = ti.field(ti.f32, 3 * self.n_verts)

        self.spring = ti.Vector.field(2, ti.i32, self.n_edges)
        self.indices = ti.field(ti.i32, 2 * self.n_edges)
        self.Jx = ti.Matrix.field(3, 3, ti.f32, self.n_edges)
        self.Jv = ti.Matrix.field(3, 3, ti.f32, self.n_edges)
        self.rest_len = ti.field(ti.f32, self.n_edges)
        self.spring_ks = ti.field(ti.f32, self.n_edges)

        # Global simulation parameters.
        self.gravity = ti.Vector([0, -9, 0])
        self.kf = 1e7
        self.kd = 0.5

        # Initialize fields from NumPy arrays.
        self._init_taichi_fields()

        # Build mass matrix.
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(3 * self.n_verts, 3 * self.n_verts, max_num_triplets=100000)
        self._init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()

        self.DBuilder = ti.linalg.SparseMatrixBuilder(3 * self.n_verts, 3 * self.n_verts, max_num_triplets=300000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(3 * self.n_verts, 3 * self.n_verts, max_num_triplets=300000)

        # Build fixed (pinned) vertices: pin top layer (j==N) in original grid.
        self._build_fixed_vertices(active_3d)

    def _build_active_vertices(self, active_3d):
        """
        Reindex the active vertices. Compute positions using the same mapping as init_positions.
        """
        N = self.N
        grid_size = N + 1
        old2new = -np.ones((grid_size, grid_size, grid_size), dtype=np.int32)
        pos_list = []
        new2old_list = []
        # Use same position formula as before.
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if active_3d[i, j, k]:
                        old2new[i, j, k] = len(pos_list)
                        x = 0.25 + 0.5 * (i / N)
                        y = 0.75 + 0.5 * (j / N)
                        z = 0.25 + 0.5 * (k / N)
                        pos_list.append([x, y, z])
                        new2old_list.append((i, j, k))
        self.pos_np = np.array(pos_list, dtype=np.float32)
        self.old2new = old2new
        self.new2old_list = new2old_list

    def _build_edges_structural(self, active_3d):
        """
        Build structural edges between active vertices: (i,j,k)->(i+1,j,k), (i,j+1,k), (i,j,k+1)
        """
        N = self.N
        grid_size = N + 1
        edges = []
        def idx(i, j, k):
            return self.old2new[i, j, k]
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if not active_3d[i, j, k]:
                        continue
                    v1 = idx(i, j, k)
                    if i < N and active_3d[i+1, j, k]:
                        edges.append((v1, idx(i+1, j, k)))
                    if j < N and active_3d[i, j+1, k]:
                        edges.append((v1, idx(i, j+1, k)))
                    if k < N and active_3d[i, j, k+1]:
                        edges.append((v1, idx(i, j, k+1)))
        self.edges_np = np.array(edges, dtype=np.int32)

    def _init_taichi_fields(self):
        """Initialize positions, velocities, mass, and spring fields from NumPy data."""
        self.pos.from_numpy(self.pos_np)
        self.initPos.from_numpy(self.pos_np)
        vel_init = np.zeros((self.pos_np.shape[0], 3), dtype=np.float32)
        force_init = np.zeros((self.pos_np.shape[0], 3), dtype=np.float32)
        mass_init = np.full((self.pos_np.shape[0],), 0.2, dtype=np.float32)
        self.vel.from_numpy(vel_init)
        self.force.from_numpy(force_init)
        self.mass.from_numpy(mass_init)
        self.spring.from_numpy(self.edges_np)

        # Set rest lengths and spring stiffness from positions.
        n_edges = self.edges_np.shape[0]
        rest_list = np.zeros((n_edges,), dtype=np.float32)
        ks_list   = np.zeros((n_edges,), dtype=np.float32)
        for e in range(n_edges):
            i1, i2 = self.edges_np[e]
            p1 = self.pos_np[i1]
            p2 = self.pos_np[i2]
            rest_list[e] = np.linalg.norm(p2 - p1)
            ks_list[e] = 100.0 + 50.0 * np.random.rand()
        self.rest_len.from_numpy(rest_list)
        self.spring_ks.from_numpy(ks_list)

    @ti.kernel
    def _init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
        for i in range(self.n_verts):
            m = self.mass[i]
            for c in ti.static(range(3)):
                M[3*i+c, 3*i+c] += m

    def _build_fixed_vertices(self, active_3d):
        """
        Build a list of fixed (pinned) vertices corresponding to the top layer (j==N) in the original grid.
        """
        N = self.N
        fix_list = []
        for new_id, (i, j, k) in enumerate(self.new2old_list):
            if j == N:
                fix_list.append(new_id)
        self.fix_vertex_list = fix_list
        self.num_fixed_vertices = len(fix_list)
        self.Jf = ti.Matrix.field(3, 3, ti.f32, self.num_fixed_vertices)

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def compute_force(self):
        # Gravity.
        for i in range(self.n_verts):
            self.force[i] += self.gravity * self.mass[i]
        # Structural spring forces.
        for e in range(self.n_edges):
            idx1 = self.spring[e][0]
            idx2 = self.spring[e][1]
            p1 = self.pos[idx1]
            p2 = self.pos[idx2]
            diff = p2 - p1
            l = diff.norm()
            if l > 1e-12:
                dir = diff / l
                stretch = l - self.rest_len[e]
                f = self.spring_ks[e] * stretch * dir
                self.force[idx1] += f
                self.force[idx2] -= f
        # Pinned vertices (top layer).
        for i in ti.static(range(self.num_fixed_vertices)):
            v = self.fix_vertex_list[i]
            self.force[v] += self.kf * (self.initPos[v] - self.pos[v])

        # Note: Additional forces (like self-collision) can be added here.

    @ti.kernel
    def compute_delta(self, h: ti.f32, plane_y: ti.f32, alpha: ti.f32):
        """
        Compute the collision correction term delta.
        For a ground plane at y = plane_y, if a particle penetrates (pos.y < plane_y),
        compute a correction delta = alpha*(penetration/h) along the positive y direction.
        The delta vector is stored in the flattened self.delta array.
        """
        for i in range(self.n_verts):
            d = ti.Vector([0.0, 0.0, 0.0])
            if self.pos[i].y < plane_y:
                penetration = plane_y - self.pos[i].y
                d = alpha * (penetration / h) * ti.Vector([0.0, 1.0, 0.0])
            self.delta[3*i+0] = d[0]
            self.delta[3*i+1] = d[1]
            self.delta[3*i+2] = d[2]

    @ti.kernel
    def compute_Jacobians(self):
        I3 = ti.Matrix.identity(ti.f32, 3)
        for e in range(self.n_edges):
            idx1 = self.spring[e][0]
            idx2 = self.spring[e][1]
            dx = self.pos[idx1] - self.pos[idx2]
            l = dx.norm()
            inv_l = 1.0 / l if l > 1e-12 else 0.0
            dxdxT = dx.outer_product(dx)
            factor = self.rest_len[e] * inv_l
            self.Jx[e] = self.spring_ks[e] * (I3 - factor * (I3 - dxdxT * (inv_l**2)))
            self.Jv[e] = self.kd * I3

        for i in ti.static(range(self.num_fixed_vertices)):
            self.Jf[i] = -self.kf * ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        for e in range(self.n_edges):
            v1 = self.spring[e][0]
            v2 = self.spring[e][1]
            J = self.Jx[e]
            for r, c in ti.static(ti.ndrange(3, 3)):
                K[3*v1+r, 3*v1+c] -= J[r, c]
                K[3*v1+r, 3*v2+c] += J[r, c]
                K[3*v2+r, 3*v1+c] += J[r, c]
                K[3*v2+r, 3*v2+c] -= J[r, c]
        for i in ti.static(range(self.num_fixed_vertices)):
            v = self.fix_vertex_list[i]
            Jf_ = self.Jf[i]
            for r, c in ti.static(ti.ndrange(3, 3)):
                K[3*v+r, 3*v+c] += Jf_[r, c]

    @ti.kernel
    def assemble_D(self, D: ti.types.sparse_matrix_builder()):
        for e in range(self.n_edges):
            v1 = self.spring[e][0]
            v2 = self.spring[e][1]
            Jv_ = self.Jv[e]
            for r, c in ti.static(ti.ndrange(3, 3)):
                D[3*v1+r, 3*v1+c] -= Jv_[r, c]
                D[3*v1+r, 3*v2+c] += Jv_[r, c]
                D[3*v2+r, 3*v1+c] += Jv_[r, c]
                D[3*v2+r, 3*v2+c] -= Jv_[r, c]

    @ti.kernel
    def updatePosVel(self, h: ti.f32, dv: ti.types.ndarray()):
        for i in range(self.n_verts):
            vx = dv[3*i+0]
            vy = dv[3*i+1]
            vz = dv[3*i+2]
            self.vel[i] += ti.Vector([vx, vy, vz])
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def copy_to(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.n_verts):
            des[3*i+0] = source[i][0]
            des[3*i+1] = source[i][1]
            des[3*i+2] = source[i][2]

    @ti.kernel
    def compute_b_collision(self, 
                            b: ti.types.ndarray(), 
                            f: ti.types.ndarray(), 
                            Kv: ti.types.ndarray(), 
                            h: ti.f32):
        for i in range(3 * self.n_verts):
            b[i] = h * (f[i] + Kv[i] * h) + h * self.delta[i]

    def spring2indices(self):
        # For line rendering.
        inds = np.zeros((2 * self.n_edges,), dtype=np.int32)
        for e in range(self.n_edges):
            inds[2*e+0] = self.edges_np[e, 0]
            inds[2*e+1] = self.edges_np[e, 1]
        self.indices.from_numpy(inds)

    def update(self, h):
        self.compute_force()
        self.compute_Jacobians()
        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()
        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()
        A = self.M - h * D - (h**2) * K
        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)
        Kv = K @ self.vel_1D
        # Compute collision correction delta (e.g., for ground collision at y=0)
        alpha = 0.1  # tuning parameter; 1.0 = full correction
        self.compute_delta(h, 0.0, alpha)
        self.compute_b_collision(self.b, self.force_1D, Kv, h)
        """
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        dv = solver.solve(self.b)"""
        solver = ti.linalg.SparseCG(A, self.b)
        dv, _ = solver.solve()

        self.updatePosVel(h, dv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="cpu", type=str, help="Backend (cpu or cuda)")
    parser.add_argument("-n", "--size", default=8, type=int, help="Grid size N")
    args = parser.parse_args()

    if args.arch in ["cpu", "x64", "arm64"]:
        ti.init(arch=ti.cpu)
    elif args.arch in ["cuda", "gpu"]:
        ti.init(arch=ti.cuda)
    else:
        raise ValueError("Only CPU/CUDA supported.")

    N = args.size
    active_3d = generate_layered_shape_shrink(N=N, p_shrink=0.3)
    cube_3d = Cube3D(N, active_3d)
    cube_3d.spring2indices()

    window = ti.ui.Window("Layered Shrink 3D Cloth with Collision Correction", (800, 800))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    dt = 0.01

    while window.running:
        cube_3d.update(dt)
        camera.position(1.2, 1.8, 2.2)
        camera.lookat(0.5, 1.0, 0.5)
        scene.set_camera(camera)
        scene.lines(cube_3d.pos, indices=cube_3d.indices, color=(0, 0, 1), width=0.002)
        scene.particles(cube_3d.pos, radius=0.005, color=(1, 0, 0))
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
