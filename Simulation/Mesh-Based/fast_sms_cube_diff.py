import argparse
import numpy as np
import taichi as ti

@ti.data_oriented
class Cube3D:
    def __init__(self, N=10):
        """
        Creates a 3D grid with (N+1)^3 vertices and only structural edges (x,y,z).
        Now with differentiable spring stiffness parameters.
        """
        self.N = N
        self.NV = (N + 1)**3
        self.NE = 3 * N * (N + 1) * (N + 1)

        # Fields for positions, velocities, etc.
        self.pos = ti.Vector.field(3, ti.f32, self.NV)
        self.vel = ti.Vector.field(3, ti.f32, self.NV)
        self.initPos = ti.Vector.field(3, ti.f32, self.NV)
        self.prevPos = ti.Vector.field(3, ti.f32, self.NV)
        self.force_ext = ti.Vector.field(3, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)

        # DIFF: Spring stiffness with gradient enabled
        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.rest_len = ti.field(ti.f32, self.NE)
        self.spring_ks = ti.field(ti.f32, self.NE, needs_grad=True)  # Gradient enabled
        self.indices = ti.field(ti.i32, shape=2 * self.NE)

        # Gravity and fixed settings
        self.gravity = ti.Vector([0, -9, 0])
        self.kf = 1e10

        # Initialization
        self.init_positions()
        self.init_edges()
        self.init_spring_stiffness()
        self.init_prev_positions()

        # DIFF: Moved matrix building to separate method
        self.rebuild_system_matrices()

        # Identify fixed vertices (top layer) and build penalty matrix.
        self.fix_vertex_ids = self.find_top_layer_vertices()
        self.PenaltyBuilder = ti.linalg.SparseMatrixBuilder(3 * self.NV, 3 * self.NV, max_num_triplets=500000)
        self.init_penalty_sp(self.PenaltyBuilder)
        self.penalty = self.PenaltyBuilder.build()

        # Flatten fields for the solver.
        self.solver_vec_size = 3 * self.NV
        self.x_1D = ti.ndarray(ti.f32, self.solver_vec_size)
        self.b_1D = ti.ndarray(ti.f32, self.solver_vec_size)
        self.f_ext_1D = ti.ndarray(ti.f32, self.solver_vec_size)
        self.y_1D = ti.ndarray(ti.f32, self.solver_vec_size)
        self.oldPos_1D = ti.ndarray(ti.f32, self.solver_vec_size)

        self.d_1D = ti.field(ti.f32, shape=(3 * self.NE))
        self.Jd_field = ti.field(ti.f32, shape=(self.solver_vec_size))
        self.rhs_field = ti.field(ti.f32, shape=(self.solver_vec_size))

    # DIFF: rebuild matrices when stiffness changes
    def rebuild_system_matrices(self):
        # Build mass matrix
        self.M_builder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NV, 10000)
        self.build_mass_matrix(self.M_builder)
        self.M = self.M_builder.build()

        # Build stiffness matrix
        self.L_builder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NV, max_num_triplets=1000000)
        self.build_laplacian(self.L_builder)
        self.L = self.L_builder.build()

        # Build coupling matrix
        self.J_builder = ti.linalg.SparseMatrixBuilder(3*self.NV, 3*self.NE, max_num_triplets=1000000)
        self.build_J(self.J_builder)
        self.J = self.J_builder.build()

    

    # -------------------------------------------
    # Initialization and build kernels
    # -------------------------------------------
    @ti.kernel
    def init_positions(self):
        for i, j, k in ti.ndrange(self.N+1, self.N+1, self.N+1):
            idx = i*(self.N+1)*(self.N+1) + j*(self.N+1) + k
            x_coord = 0.25 + 0.5 * (i / self.N)
            y_coord = 0.75 + 0.5 * (j / self.N)
            z_coord = 0.25 + 0.5 * (k / self.N)
            self.pos[idx] = ti.Vector([x_coord, y_coord, z_coord])
            self.initPos[idx] = self.pos[idx]
            self.vel[idx] = ti.Vector.zero(ti.f32, 3)
            self.mass[idx] = 1

    @ti.kernel
    def init_edges(self):
        e = 0
        for i, j, k in ti.ndrange(self.N+1, self.N+1, self.N+1):
            idx = i*(self.N+1)*(self.N+1) + j*(self.N+1) + k
            # x-direction
            if i < self.N:
                idx2 = (i+1)*(self.N+1)*(self.N+1) + j*(self.N+1) + k
                self.spring[e] = ti.Vector([idx, idx2])
                self.rest_len[e] = (self.pos[idx] - self.pos[idx2]).norm()
                e += 1
            # y-direction
            if j < self.N:
                idx2 = i*(self.N+1)*(self.N+1) + (j+1)*(self.N+1) + k
                self.spring[e] = ti.Vector([idx, idx2])
                self.rest_len[e] = (self.pos[idx] - self.pos[idx2]).norm()
                e += 1
            # z-direction
            if k < self.N:
                idx2 = i*(self.N+1)*(self.N+1) + j*(self.N+1) + (k+1)
                self.spring[e] = ti.Vector([idx, idx2])
                self.rest_len[e] = (self.pos[idx] - self.pos[idx2]).norm()
                e += 1

    @ti.kernel
    def init_spring_stiffness(self):
        for e in range(self.NE):
            self.spring_ks[e] = 100.0


    @ti.kernel
    def init_prev_positions(self):
        for i in range(self.NV):
            self.prevPos[i] = self.pos[i]

    @ti.kernel
    def build_mass_matrix(self, Mb: ti.types.sparse_matrix_builder()):
        for i in range(self.NV):
            mi = self.mass[i]
            for c in ti.static(range(3)):
                Mb[3 * i + c, 3 * i + c] += mi

    @ti.kernel
    def build_laplacian(self, Lb: ti.types.sparse_matrix_builder()):
        for e in range(self.NE):
            i1, i2 = self.spring[e][0], self.spring[e][1]
            k_e = self.spring_ks[e]
            for c in ti.static(range(3)):
                Lb[3 * i1 + c, 3 * i1 + c] += k_e
                Lb[3 * i2 + c, 3 * i2 + c] += k_e
                Lb[3 * i1 + c, 3 * i2 + c] -= k_e
                Lb[3 * i2 + c, 3 * i1 + c] -= k_e

    @ti.kernel
    def build_J(self, Jb: ti.types.sparse_matrix_builder()):
        """
        Build coupling matrix J = for each spring e connecting vertices i1 and i2:
          For each component c:
            J[3*i1+c, 3*e+c] += k_e
            J[3*i2+c, 3*e+c] -= k_e
        """
        for e in range(self.NE):
            i1 = self.spring[e][0]
            i2 = self.spring[e][1]
            k_e = self.spring_ks[e]
            for c in ti.static(range(3)):
                Jb[3 * i1 + c, 3 * e + c] += k_e
                Jb[3 * i2 + c, 3 * e + c] -= k_e
    
    def find_top_layer_vertices(self):
        fixed_ids = []
        for i in range(self.N + 1):
            for k in range(self.N + 1):
                j = self.N
                idx = i * (self.N + 1) * (self.N + 1) + j * (self.N + 1) + k
                fixed_ids.append(idx)
        return fixed_ids

    @ti.kernel
    def init_penalty_sp(self, P: ti.types.sparse_matrix_builder()):
        for v in ti.static(self.fix_vertex_ids):
            for c in ti.static(range(3)):
                P[3 * v + c, 3 * v + c] += self.kf

    @ti.kernel
    def clear_ext_forces(self):
        for i in range(self.NV):
            self.force_ext[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def add_gravity(self):
        for i in range(self.NV):
            self.force_ext[i] = self.mass[i] * self.gravity

    @ti.kernel
    def spring2indices(self):
        for i in range(self.NE):
            self.indices[2 * i + 0] = self.spring[i][0]
            self.indices[2 * i + 1] = self.spring[i][1]

    @ti.kernel
    def copy_pos_to_nd(self, arr: ti.types.ndarray()):
        for i in range(self.NV):
            for c in ti.static(range(3)):
                arr[3 * i + c] = self.pos[i][c]

    @ti.kernel
    def copy_nd_to_pos(self, arr: ti.types.ndarray()):
        for i in range(self.NV):
            self.pos[i] = ti.Vector([arr[3 * i + 0],
                                     arr[3 * i + 1],
                                     arr[3 * i + 2]])

    @ti.kernel
    def copy_force_to_nd(self, arr: ti.types.ndarray()):
        for i in range(self.NV):
            for c in ti.static(range(3)):
                arr[3 * i + c] = self.force_ext[i][c]

    @ti.kernel
    def compute_y(self, y_arr: ti.types.ndarray()):
        """
        Compute y = 2*q_n - q_{n-1} for each vertex.
        """
        for i in range(self.NV):
            for c in ti.static(range(3)):
                y_arr[3 * i + c] = 2 * self.pos[i][c] - self.prevPos[i][c]

    @ti.kernel
    def compute_rhs_corrected(self, y_arr: ti.types.ndarray(), f_arr: ti.types.ndarray(), h: ti.f32, b_arr: ti.types.ndarray()):
        """
        Compute the inertial-plus-external term:
            b = - M * y + h^2 * f_ext,
        where y = 2*q_n - q_{n-1}.
        """
        for i in range(self.NV):
            for c in ti.static(range(3)):
                b_arr[3 * i + c] = - self.mass[i] * y_arr[3 * i + c] + (h**2) * f_arr[3 * i + c]

    @ti.kernel
    def compute_d(self):
        for e in range(self.NE):
            i1 = self.spring[e][0]
            i2 = self.spring[e][1]
            diff = self.pos[i1] - self.pos[i2]
            norm = diff.norm()
            if norm > 1e-8:
                for c in ti.static(range(3)):
                    self.d_1D[3 * e + c] = self.rest_len[e] * diff[c] / norm
            else:
                for c in ti.static(range(3)):
                    self.d_1D[3 * e + c] = 0.0

    @ti.kernel
    def updateVelocities_from_solution(self, x_arr: ti.types.ndarray(), y_arr: ti.types.ndarray(), h: ti.f32):
        for i in range(self.NV):
            for c in ti.static(range(3)):
                self.vel[i][c] = (x_arr[3 * i + c] - y_arr[3 * i + c]) / h

    @ti.kernel
    def copy_pos_to_prevPos(self, pos_arr: ti.types.ndarray()):
        for i in range(self.NV):
            self.prevPos[i] = ti.Vector([pos_arr[3 * i + 0],
                                         pos_arr[3 * i + 1],
                                         pos_arr[3 * i + 2]])

    #
    # New kernels to keep everything as fields until we flatten at step 10
    #
    @ti.kernel
    def compute_Jd_field(self):
        """
        Manually compute J*d in field form. 
        Jd_field[i] = sum_{edges e touching vertex i} ( +/- k_e * d[e] ).
        """
        # Clear Jd_field first
        for i in range(self.solver_vec_size):
            self.Jd_field[i] = 0.0

        for e in range(self.NE):
            i1 = self.spring[e][0]
            i2 = self.spring[e][1]
            ke = self.spring_ks[e]
            for c in ti.static(range(3)):
                self.Jd_field[3 * i1 + c] += ke * self.d_1D[3 * e + c]
                self.Jd_field[3 * i2 + c] -= ke * self.d_1D[3 * e + c]

    @ti.kernel
    def compute_final_rhs_field(self, h: ti.f32, b_arr: ti.types.ndarray()):
        """
        final RHS = h^2 * Jd_field - b_1D
        We'll store the result in self.rhs_field.
        """
        for i in range(self.solver_vec_size):
            self.rhs_field[i] = (h**2) * self.Jd_field[i] - b_arr[i]

    @ti.kernel
    def copy_field_to_ndarray(self, src: ti.template(), dst: ti.types.ndarray()):
        """
        Copy a 1D field of length solver_vec_size into an NDArray of the same length.
        """
        for i in range(self.solver_vec_size):
            dst[i] = src[i]
    

    # DIFF:  loss function measuring vertical displacement
    @ti.kernel
    def compute_loss(self) -> ti.f32:
        loss = 0.0
        for i in range(self.NV):
            # Penalize vertical displacement from initial position
            loss += (self.pos[i].y - self.initPos[i].y)**2
        return loss
    
    
    def update(self, h):
        # DIFF: Wrap simulation step in autograd tape
        with ti.ad.Tape(loss=self.compute_loss()):
            # DIFF: Rebuild matrices with current stiffness values
            self.rebuild_system_matrices()

            # Original simulation logic
            self.clear_ext_forces()
            self.add_gravity()
            self.copy_pos_to_nd(self.oldPos_1D)
            self.compute_y(self.y_1D)
            A = self.M + (h**2) * (self.L + self.penalty)
            self.copy_force_to_nd(self.f_ext_1D)
            self.compute_rhs_corrected(self.y_1D, self.f_ext_1D, h, self.b_1D)
            self.compute_d()
            self.compute_Jd_field()
            self.compute_final_rhs_field(h, self.b_1D)
            
            rhs_temp = ti.ndarray(ti.f32, shape=(self.solver_vec_size))
            self.copy_field_to_ndarray(self.rhs_field, rhs_temp)
            
            solver = ti.linalg.SparseSolver(solver_type="LLT")
            solver.analyze_pattern(A)
            solver.factorize(A)
            x_new = solver.solve(rhs_temp)

            self.updateVelocities_from_solution(x_new, self.y_1D, h)
            self.copy_pos_to_prevPos(self.oldPos_1D)
            self.copy_nd_to_pos(x_new)

        # DIFF: After backward pass, gradients are available
        print("Max stiffness gradient:", ti.max(self.spring_ks.grad))
        print("Min stiffness gradient:", ti.min(self.spring_ks.grad))

    def display(self, gui, radius=3, color=0xFFFFFF):
        lines = self.spring.to_numpy()
        pos_np = self.pos.to_numpy()
        edge_begin = np.zeros((lines.shape[0], 2), dtype=np.float32)
        edge_end = np.zeros((lines.shape[0], 2), dtype=np.float32)
        for i in range(lines.shape[0]):
            idx1, idx2 = lines[i]
            x1, y1, z1 = pos_np[idx1]
            x2, y2, z2 = pos_np[idx2]
            edge_begin[i] = [x1, z1]
            edge_end[i] = [x2, z2]
        gui.lines(edge_begin, edge_end, radius=1, color=0x0000FF)
        circles_pos = np.array([[p[0], p[2]] for p in pos_np])
        gui.circles(circles_pos, radius=radius, color=color)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--use-ggui", action="store_true", help="Use GGUI")
    parser.add_argument("-a", "--arch", default="cpu", type=str, help="Backend")
    args = parser.parse_args()

    ti.init(arch=ti.cpu if args.arch in ["cpu", "x64", "arm64"] else ti.cuda,
            default_fp=ti.f32)  # DIFF: Set default precision for autograd

    cube_3d = Cube3D(N=8)  # Smaller grid for demonstration
    h = 0.03
    cube_3d.spring2indices()

    # DIFF: Initialize stiffness gradients
    cube_3d.spring_ks.grad.fill(0.0)

    window = ti.ui.Window("Differentiable 3D Cloth", (800, 800), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene() 
    canvas.set_background_color((1, 1, 1))
    camera = ti.ui.Camera()
    
    while window.running:
        cube_3d.update(h)
        camera.position(1.2, 1.2, 1.2)
        camera.lookat(0.5, 0.5, 0.5)
        scene.set_camera(camera)
        scene.lines(cube_3d.pos, indices=cube_3d.indices, color=(0, 0, 1), width=0.002)
        scene.particles(cube_3d.pos, radius=0.005, color=(0, 0, 1))
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()