# use the collision detection and the opposite force 
import argparse
import numpy as np
import taichi as ti

def generate_layered_shape_shrink(N=8, p_shrink=0.3):
    """
    Generate a layered solid shape on an (N+1)x(N+1)x(N+1) grid with:
      1. A randomly picked starting layer j0 (1 <= j0 < N).
      2. Upward/downward layers with boundaries that randomly shrink 
         (with probability p_shrink each time).
    Returns:
      active_3d: boolean array shape=(N+1, N+1, N+1), 
                 active_3d[i,j,k] == True if vertex is included.
    """
    grid_size = N+1
    active_3d = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    j0 = np.random.randint(1, N)
    print(f"Chosen starting layer: {j0}")
    
    # For N=8, define an 8×8 block on layer j0:
    i_min0, i_max0 = 1, grid_size   # i in [1..8]
    k_min0, k_max0 = 1, grid_size   # k in [1..8]
    for i in range(i_min0, i_max0):
        for k in range(k_min0, k_max0):
            active_3d[i, j0, k] = True
    
    def shrink_bound(i_min, i_max, k_min, k_max):
        # randomly shrink boundaries
        if np.random.rand() < p_shrink:
            i_min += 1
        if np.random.rand() < p_shrink:
            i_max -= 1
        if np.random.rand() < p_shrink:
            k_min += 1
        if np.random.rand() < p_shrink:
            k_max -= 1
        # ensure region remains >= 2×2
        if (i_max - i_min) < 2:
            i_min, i_max = i_min0, i_max0
        if (k_max - k_min) < 2:
            k_min, k_max = k_min0, k_max0
        return i_min, i_max, k_min, k_max

    # Upwards: j0+1..N
    curr_i_min, curr_i_max = i_min0, i_max0
    curr_k_min, curr_k_max = k_min0, k_max0
    for j in range(j0+1, grid_size):
        curr_i_min, curr_i_max, curr_k_min, curr_k_max = shrink_bound(
            curr_i_min, curr_i_max, curr_k_min, curr_k_max
        )
        for i in range(curr_i_min, curr_i_max):
            for k in range(curr_k_min, curr_k_max):
                active_3d[i,j,k] = True

    # Downwards: j0-1..0
    curr_i_min, curr_i_max = i_min0, i_max0
    curr_k_min, curr_k_max = k_min0, k_max0
    for j in range(j0-1, -1, -1):
        curr_i_min, curr_i_max, curr_k_min, curr_k_max = shrink_bound(
            curr_i_min, curr_i_max, curr_k_min, curr_k_max
        )
        for i in range(curr_i_min, curr_i_max):
            for k in range(curr_k_min, curr_k_max):
                active_3d[i,j,k] = True

    return active_3d


@ti.data_oriented
class Cube3D:
    def __init__(self, N, active_3d):
        """
        Create a "3D structural cloth" but only for the subset of vertices
        specified by 'active_3d[i,j,k]==True'.
        
        Steps:
          1) Reindex active vertices => [0..n_active-1].
          2) Build pos, mass arrays of size n_active.
          3) Build structural edges only among active neighbors.
          4) Pin the top layer j=N if that vertex is active.
        """
        self.N = N
        # (A) Reindex active vertices
        self._build_active_vertices(active_3d)

        # (B) Build structural edges among active neighbors
        self._build_edges_structural(active_3d)

        # 3D fields for solver
        self.vel = ti.Vector.field(3, ti.f32, self.n_verts)
        self.force = ti.Vector.field(3, ti.f32, self.n_verts)
        self.initPos = ti.Vector.field(3, ti.f32, self.n_verts)
        self.mass = ti.field(ti.f32, self.n_verts)

        self.vel_1D = ti.ndarray(ti.f32, 3*self.n_verts)
        self.force_1D = ti.ndarray(ti.f32, 3*self.n_verts)
        self.b = ti.ndarray(ti.f32, 3*self.n_verts, needs_grad=True)

        # Build these as taichi fields
        self.pos = ti.Vector.field(3, ti.f32, self.n_verts)
        self.spring = ti.Vector.field(2, ti.i32, self.n_edges)
        self.indices = ti.field(ti.i32, 2*self.n_edges)
        self.Jx = ti.Matrix.field(3,3, ti.f32, self.n_edges)
        self.Jv = ti.Matrix.field(3,3, ti.f32, self.n_edges)
        self.rest_len = ti.field(ti.f32, self.n_edges)
        self.spring_ks = ti.field(ti.f32, self.n_edges)

        # Global parameters
        self.gravity = ti.Vector([0, -9, 0])
        self.kf = 1e7
        self.kd = 0.5

        # (C) Copy data into Taichi fields
        self._init_taichi_fields()

        # (D) Build mass matrix
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(3*self.n_verts, 3*self.n_verts, max_num_triplets=100000)
        self._init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()

        # D, K builders
        self.DBuilder = ti.linalg.SparseMatrixBuilder(3*self.n_verts, 3*self.n_verts, max_num_triplets=300000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(3*self.n_verts, 3*self.n_verts, max_num_triplets=300000)

        # (E) Build pinned top layer (only if active, which it is if we reindexed it)
        self._build_fixed_vertices(active_3d)

    def _build_active_vertices(self, active_3d):
        """
        1) Figure out how many are active.
        2) Create old2new map, new2old map.
        3) Store pos_np.
        """
        N = self.N
        old2new = -np.ones(((N+1), (N+1), (N+1)), dtype=np.int32)

        def gid(i,j,k):
            return i*(N+1)*(N+1) + j*(N+1) + k

        # Step A: count how many are active
        counter = 0
        # We'll also store positions in a Python list
        pos_list = []
        new2old_list = []

        for i in range(N+1):
            for j in range(N+1):
                for k in range(N+1):
                    if active_3d[i,j,k]:
                        old2new[i,j,k] = counter
                        # compute position (same logic as your init_positions)
                        x = 0.25 + 0.5*(i/N)
                        y = 0.75 + 0.5*(j/N)
                        z = 0.25 + 0.5*(k/N)
                        pos_list.append([x,y,z])
                        new2old_list.append( (i,j,k) )
                        counter += 1

        self.n_verts = counter
        self.pos_np = np.array(pos_list, dtype=np.float32)  # shape (n_verts, 3)
        self.old2new = old2new
        self.new2old_list = new2old_list  # array of (i,j,k) for each new ID

    def _build_edges_structural(self, active_3d):
        """
        Structural edges:
          (i,j,k) -> (i+1,j,k), (i,j+1,k), (i,j,k+1),
        but only if both endpoints are active.
        """
        N = self.N
        edges = []
        rest = []  # store rest length
        def idx(i,j,k):
            return self.old2new[i,j,k]

        for i in range(N+1):
            for j in range(N+1):
                for k in range(N+1):
                    if not active_3d[i,j,k]:
                        continue
                    v1 = idx(i,j,k)
                    # x-dir
                    if i<N and active_3d[i+1,j,k]:
                        v2 = idx(i+1,j,k)
                        edges.append((v1,v2))
                    # y-dir
                    if j<N and active_3d[i,j+1,k]:
                        v2 = idx(i,j+1,k)
                        edges.append((v1,v2))
                    # z-dir
                    if k<N and active_3d[i,j,k+1]:
                        v2 = idx(i,j,k+1)
                        edges.append((v1,v2))

        self.edges_np = np.array(edges, dtype=np.int32)  # shape (n_edges, 2)
        self.n_edges = self.edges_np.shape[0]

    def _init_taichi_fields(self):
        """
        Fill self.pos, self.spring, self.rest_len, self.spring_ks, etc.
        """
        # 1) Positions
        self.pos.from_numpy(self.pos_np)

        # 2) Edges
        self.spring.from_numpy(self.edges_np)

        # 3) Rest lengths & ks
        rest_list = np.zeros((self.n_edges,), dtype=np.float32)
        ks_list   = np.zeros((self.n_edges,), dtype=np.float32)

        # compute rest length from positions
        for e in range(self.n_edges):
            i1, i2 = self.edges_np[e]
            p1 = self.pos_np[i1]
            p2 = self.pos_np[i2]
            dist = np.linalg.norm(p2 - p1)
            rest_list[e] = dist
            ks_list[e] = 1000.0 + 500.0*np.random.rand()  # example random stiffness

        self.rest_len.from_numpy(rest_list)
        self.spring_ks.from_numpy(ks_list)

        # 4) We'll also keep track of velocities, force, mass in NumPy for init
        vel_init = np.zeros((self.n_verts,3), dtype=np.float32)
        force_init = np.zeros((self.n_verts,3), dtype=np.float32)
        mass_init = np.full((self.n_verts,), 0.2, dtype=np.float32)  # default mass = 0.2

        self.vel.from_numpy(vel_init)
        self.force.from_numpy(force_init)
        self.mass.from_numpy(mass_init)
        self.initPos.from_numpy(self.pos_np)

    @ti.kernel
    def _init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
        for i in range(self.n_verts):
            m = self.mass[i]
            for c in ti.static(range(3)):
                M[3*i + c, 3*i + c] += m

    def _build_fixed_vertices(self, active_3d):
        """
        Pin the top layer (j=N) if it is active in old space.
        We'll find all new IDs that correspond to j=N.
        """
        fix_list = []
        for new_id, (i,j,k) in enumerate(self.new2old_list):
            if j == self.N:
                fix_list.append(new_id)
        self.fix_vertex_list = fix_list
        self.num_fixed_vertices = len(fix_list)
        self.Jf = ti.Matrix.field(3,3, ti.f32, self.num_fixed_vertices)

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector.zero(ti.f32,3)

    @ti.func
    def compute_collision_forces(self):
        # Parameters: adjust these as needed
        collision_radius = 0.01    # radius around each particle for collision detection
        col_force_k = 5          # collision stiffness (a relatively small value)

        # (A) Particle–Particle collisions
        for i in range(self.n_verts):
            for j in range(i+1, self.n_verts):
                diff = self.pos[j] - self.pos[i]
                dist = diff.norm()
                # If distance is less than twice the collision radius (i.e., spheres overlap)
                if dist < 2 * collision_radius and dist > 1e-6:
                    penetration = 2 * collision_radius - dist
                    repulsion = col_force_k * penetration * (diff / dist)
                    # Apply equal and opposite forces
                    self.force[i] -= repulsion
                    self.force[j] += repulsion

        # (B) Particle–Edge collisions (optional)
        # For each edge, check each particle (not an endpoint) for proximity.
        for e in range(self.n_edges):
            A = self.pos[self.spring[e][0]]
            B = self.pos[self.spring[e][1]]
            AB = B - A
            AB_norm_sq = AB.dot(AB)
            for i in range(self.n_verts):
                # Skip if this particle is an endpoint of the edge.
                if i == self.spring[e][0] or i == self.spring[e][1]:
                    continue
                P = self.pos[i]
                AP = P - A
                # Compute projection parameter t in [0,1]
                t = AP.dot(AB) / (AB_norm_sq + 1e-12)
                t = ti.min(ti.max(t, 0.0), 1.0)
                closest = A + t * AB
                diff = P - closest
                d = diff.norm()
                # If the distance is less than the collision radius:
                if d < collision_radius and d > 1e-6:
                    penetration = collision_radius - d
                    repulsion = col_force_k * penetration * (diff / d)
                    # Apply repulsive force to the particle
                    self.force[i] += repulsion
                    # Optionally, you could also distribute a part of this force to the edge endpoints

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        # gravity
        for i in range(self.n_verts):
            self.force[i] += self.gravity*self.mass[i]

        # structural
        for e in range(self.n_edges):
            idx1 = self.spring[e][0]
            idx2 = self.spring[e][1]
            p1 = self.pos[idx1]
            p2 = self.pos[idx2]
            dist = p2 - p1
            length = dist.norm()
            if length>1e-12:
                dir = dist/length
                stretch = length - self.rest_len[e]
                f = self.spring_ks[e]*stretch*dir
                self.force[idx1]+=f
                self.force[idx2]-=f

        # pinned top
        for idx in ti.static(range(self.num_fixed_vertices)):
            v_id = self.fix_vertex_list[idx]
            self.force[v_id] += self.kf*(self.initPos[v_id] - self.pos[v_id])
        
        # collision forces
        self.compute_collision_forces()
    @ti.kernel
    def compute_Jacobians(self):
        I3 = ti.Matrix.identity(ti.f32,3)
        for e in range(self.n_edges):
            idx1 = self.spring[e][0]
            idx2 = self.spring[e][1]
            dx = self.pos[idx1] - self.pos[idx2]
            l = dx.norm()
            inv_l = 1.0/l if l>1e-12 else 0.0
            dxdxT = dx.outer_product(dx)
            factor = self.rest_len[e]*inv_l
            self.Jx[e] = self.spring_ks[e]*( I3 - factor*(I3 - dxdxT*(inv_l**2)) )
            self.Jv[e] = self.kd*I3

        # pinned
        for i in ti.static(range(self.num_fixed_vertices)):
            self.Jf[i] = -self.kf * ti.Matrix.identity(ti.f32,3)

    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        for e in range(self.n_edges):
            v1 = self.spring[e][0]
            v2 = self.spring[e][1]
            J = self.Jx[e]
            for r,c in ti.static(ti.ndrange(3,3)):
                K[3*v1 + r, 3*v1 + c] -= J[r,c]
                K[3*v1 + r, 3*v2 + c] += J[r,c]
                K[3*v2 + r, 3*v1 + c] += J[r,c]
                K[3*v2 + r, 3*v2 + c] -= J[r,c]

        # pinned
        for i in ti.static(range(self.num_fixed_vertices)):
            v_id = self.fix_vertex_list[i]
            Jf_ = self.Jf[i]
            for r,c in ti.static(ti.ndrange(3,3)):
                K[3*v_id+r, 3*v_id+c]+= Jf_[r,c]

    @ti.kernel
    def assemble_D(self, D: ti.types.sparse_matrix_builder()):
        for e in range(self.n_edges):
            v1 = self.spring[e][0]
            v2 = self.spring[e][1]
            Jd = self.Jv[e]
            for r,c in ti.static(ti.ndrange(3,3)):
                D[3*v1+r, 3*v1+c] -= Jd[r,c]
                D[3*v1+r, 3*v2+c] += Jd[r,c]
                D[3*v2+r, 3*v1+c] += Jd[r,c]
                D[3*v2+r, 3*v2+c] -= Jd[r,c]

    @ti.kernel
    def updatePosVel(self, h: ti.f32, dv: ti.types.ndarray()):
        for i in range(self.n_verts):
            vx = dv[3*i+0]
            vy = dv[3*i+1]
            vz = dv[3*i+2]
            self.vel[i] += ti.Vector([vx,vy,vz])
            self.pos[i] += h*self.vel[i]

    @ti.kernel
    def copy_to(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.n_verts):
            des[3*i+0] = source[i][0]
            des[3*i+1] = source[i][1]
            des[3*i+2] = source[i][2]

    @ti.kernel
    def compute_b(self, b: ti.types.ndarray(),
                f: ti.types.ndarray(),
                Kv: ti.types.ndarray(),
                h: ti.f32):
        for i in range(3*self.n_verts):
            b[i] = h*(f[i] + Kv[i]*h)

    def spring2indices(self):
        """
        For GGUI line drawing: store each edge in pairs [2*e, 2*e+1].
        """
        inds = np.zeros((2*self.n_edges,), dtype=np.int32)
        for e in range(self.n_edges):
            inds[2*e + 0] = self.edges_np[e,0]
            inds[2*e + 1] = self.edges_np[e,1]
        self.indices.from_numpy(inds)

    def update(self, h):
        self.compute_force()
        self.compute_Jacobians()

        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()
        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()

        A = self.M - h*D - (h**2)*K

        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)

        Kv = K @ self.vel_1D
        self.compute_b(self.b, self.force_1D, Kv, h)
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
    parser.add_argument("-g", "--use-ggui", action="store_true", help="Use GGUI")
    parser.add_argument("-a", "--arch", default="cpu", type=str, help="Backend")
    parser.add_argument("-n", "--size", default=8, type=int, help="Grid size N")
    args = parser.parse_args()

    # 1) Taichi init
    if args.arch in ["cpu","x64","arm64"]:
        ti.init(arch=ti.cpu)
    elif args.arch in ["cuda","gpu"]:
        ti.init(arch=ti.cuda)
    else:
        raise ValueError("Only CPU/CUDA supported in this snippet.")

    # 2) Generate shape
    N = args.size
    active_3d = generate_layered_shape_shrink(N=N, p_shrink=0.3)

    # 3) Build "Cube3D" from the boolean mask
    cube_3d = Cube3D(N, active_3d)
    cube_3d.spring2indices()

    # 4) GGUI
    window = ti.ui.Window("Shrinking Layer 3D Cloth", (800, 800))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))

    h = 0.01
    while window.running:
        cube_3d.update(h)

        camera.position(1.2,1.8,2.2)
        camera.lookat(0.5,1.0,0.5)
        scene.set_camera(camera)

        scene.lines(cube_3d.pos, indices=cube_3d.indices, color=(0,0,1), width=0.002)
        scene.particles(cube_3d.pos, radius=0.005, color=(1,0,0))

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()