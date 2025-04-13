# This is a testing of which part could be taken out for the automatic differentiation

import argparse
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)
@ti.data_oriented
class Cloth:
    def __init__(self, N):
        self.N = N
        self.NV = (N + 1) ** 2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # number of edges without bending edges
        self.NB = 2 * (N+1) * (N-1)  # number of bending springs
        self.NE_total = self.NE + self.NB  # total number of springs

        # -- Vertex data
        self.pos = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)

        # For sparse solver usage
        self.vel_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.force_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.b = ti.field(ti.f32, 2 * self.NV, needs_grad=True)

        # -- Spring data
        # the spring vlues are stored in indices [0, NE) for structural/shear springs
        # and [NE, NE_total) for bending:
        self.spring = ti.Vector.field(2, ti.i32, self.NE_total)
        self.indices = ti.field(ti.i32, 2 * self.NE) # for GGUI visualization
        # Jacobians for each spring, note that the Jacobians are 2x2 matrices as they are the 
        # force derivatives with respect to the position and velocity of the two **vertices**.
        self.Jx = ti.Matrix.field(2, 2, ti.f32, self.NE_total)  # Jacobian w.r.t. position
        self.Jv = ti.Matrix.field(2, 2, ti.f32, self.NE_total)  # Jacobian w.r.t. velocity
        self.rest_len = ti.field(ti.f32, self.NE_total)

        # Heterogeneous spring stiffness: each edge can have its own stiffness
        self.spring_ks = ti.field(ti.f32, self.NE, needs_grad=True)   # Heterogeneous stiffness
        self.bend_ks = ti.field(ti.f32, self.NB)  # bending stiffness

        # Global parameters (can remain global or partially heterogeneous if desired)
        self.kd = 0.5   # damping constant
        self.kf = 1.0e7 # fix point stiffness
        self.gravity = ti.Vector([0.0, -1.0])

        self.init_pos()
        self.init_edges()
        self.init_spring_stiffness()  # Initialize per-edge stiffness
        self.init_bend_stiffness()  # Initialize per-edge bending stiffness

        # Build the mass matrix (sparse)
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=1000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()

        # Additional structures for damping & stiffness in global system
        """
        Create a builder using ti.linalg.SparseMatrixBuilder().
        Call ti.kernel to fill the builder with your matrices' data.
        Build sparse matrices from the builder.
        More info: See https://docs.taichi-lang.org/docs/master/sparse_matrix"""
        self.DBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=100000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=100000)

        # Top row of the vertices are fixed
        self.fix_vertex_list = [i * (N + 1) + N for i in range(N + 1)]
        self.Jf = ti.Matrix.field(2, 2, ti.f32, len(self.fix_vertex_list))
        self.num_fixed_vertices = len(self.fix_vertex_list)

        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.pos[k] = ti.Vector([i, j]) / self.N * 0.5 + ti.Vector([0.25, 0.25])
            self.initPos[k] = self.pos[k]
            self.vel[k] = ti.Vector([0, 0])
            # Make the mass heterogeneous if desired:
            # E.g. random in [0.5, 1.5]
            self.mass[k] = 0.2   # or 1.0 + 0.5 * ti.random()

    @ti.kernel
    def init_edges(self):
        # Initialize the structural, shear, and bending indices and save them into spring.
        pos, spring, N, rest_len = ti.static(self.pos, self.spring, self.N, self.rest_len)
        # ---------------------------
        # Structural Springs (Stretch)
        # ---------------------------
        # Horizontal springs: connect (i, j) to (i, j+1)
        for i, j in ti.ndrange(N + 1, N):
            idx = i * N + j
            idx1 = i * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx1 + 1])
            rest_len[idx] = (pos[idx1] - pos[idx1 + 1]).norm()
            
        # Vertical springs: connect (i, j) to (i+1, j)
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            idx = start + i + j * N
            idx1 = i * (N + 1) + j
            idx2 = (i + 1) * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
            
        # ---------------------------
        # Shear Springs (Diagonal)
        # ---------------------------
        # Diagonal springs (first set): connect (i, j) to (i+1, j+1)
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            idx = start + i * N + j
            idx1 = i * (N + 1) + j
            idx2 = (i + 1) * (N + 1) + j + 1
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
            
        # Diagonal springs (second set): connect (i, j+1) to (i+1, j)
        start = 2 * N * (N + 1) + N * N
        for i, j in ti.ndrange(N, N):
            idx = start + i * N + j
            idx1 = i * (N + 1) + j + 1
            idx2 = (i + 1) * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
            
        # ---------------------------
        # Bending Springs
        # ---------------------------
        # Bending springs start after structural & shear springs.
        bend_offset = 2 * N * (N + 1) + 2 * N * N

        # Horizontal bending springs: connect (i, j) to (i, j+2)
        for i, j in ti.ndrange(N + 1, N - 1):
            idx = bend_offset + i * (N - 1) + j
            idx1 = i * (N + 1) + j
            idx2 = i * (N + 1) + (j + 2)
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
            
        # Vertical bending springs: connect (i, j) to (i+2, j)
        # Compute offset for vertical bending springs:
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
        """
        for e in range(self.NE):
            # randomly vary stiffness between [3e3, 6e3]:
            self.spring_ks[e] = 1000.0 + 300.0 * ti.random()
    
    @ti.kernel
    def init_bend_stiffness(self):
        # Compute the average stiffness from the structural/shear springs.
        avg = 0.0
        for e in range(self.NE):
            avg += self.spring_ks[e]
        avg /= self.NE
        # Set the bending stiffness as a fraction (say, 0.1) of the average.
        for i in range(self.NB):
            self.bend_ks[i] = 0.1 * avg


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
        # Clear forces and add gravity.
        self.clear_force()
        for i in self.force:
            self.force[i] += self.gravity * self.mass[i]

        # Structural and shear springs:
        # Use indices 0 to self.NE - 1 for structural and shear springs.
        for i in range(self.NE):
            idx1 = self.spring[i][0]
            idx2 = self.spring[i][1]
            pos1 = self.pos[idx1]
            pos2 = self.pos[idx2]
            dis = pos2 - pos1
            # Use the same method as original's compute_force:
            force = self.spring_ks[i] * (dis.norm() - self.rest_len[i]) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force

        # Bending springs:
        # These are stored in indices [self.NE, self.NE_total)
        for i in range(self.NE, self.NE_total):
            idx1 = self.spring[i][0]
            idx2 = self.spring[i][1]
            pos1 = self.pos[idx1]
            pos2 = self.pos[idx2]
            dis = pos2 - pos1
            # Retrieve the bending stiffness by indexing into self.bend_ks;
            # note: bending springs are stored starting at 0 in self.bend_ks.
            force = self.bend_ks[i - self.NE] * (dis.norm() - self.rest_len[i]) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force

        # Fixed point constraints (example: fix one vertex at index self.N and one at self.NV - 1)
        self.force[self.N] += self.kf * (self.initPos[self.N] - self.pos[self.N])
        self.force[self.NV - 1] += self.kf * (self.initPos[self.NV - 1] - self.pos[self.NV - 1])


    @ti.kernel
    def compute_Jacobians(self):
        # Iterate over all springs (structural/shear and bending)
        for i in range(self.NE_total):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1]],
                            [dx[1] * dx[0], dx[1] * dx[1]]])
            l = dx.norm()
            # Use a small threshold to avoid division by zero.
            inv_l = 1.0 / l if l > 1e-12 else 0.0
            
            # Choose the appropriate stiffness:
            k_current = 999.0# initialization
            if i < self.NE:
                # Structural/Shear spring: use per-edge stiffness from spring_ks.
                k_current = self.spring_ks[i]
            else:
                # Bending spring: use bending stiffness from bend_ks.
                k_current = self.bend_ks[i - self.NE]
            
            # Compute the Jacobian for the spring force using the linearized form.
            # print(k_current) # in practice, bend_ks are alomst the same (the mean of spring_ks)
            self.Jx[i] = (I - self.rest_len[i] * inv_l * (I - dxtdx * (inv_l**2))) * k_current
            self.Jv[i] = self.kd * I

        # Fixed point constraint Hessian: update all fixed vertices.
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
    

    """
    The damping contribution to the force (i.e. the part that comes from the velocities) is linear in the velocities. 
    For example, a spring's damping force is defined as 
    $$
    \mathbf{f}_d=-k_d\left(\mathbf{v}_1-\mathbf{v}_2\right)
    $$
    which implies that the derivative with respect to the velocities (i.e. $\frac{\partial \mathbf{f}_d}{\partial \mathbf{v}}$ ) is constant. 
    In this simple case, the derivative with respect to the velocity of vertex 1 is
    $$
    \frac{\partial \mathbf{f}_d}{\partial \mathbf{v}_1}=-k_d I
    $$
    and with respect to vertex 2 is
    $$
    \frac{\partial \mathbf{f}_d}{\partial \mathbf{v}_2}=k_d I
    $$
    where $I$ is the $2 \times 2$ identity matrix.
    """
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
        """
        The linear solvers require data in a flat, one-dimensional format rather than as a vector field. 
        Essentially, it "flattens" the 2D vector field into a 1D array where every two consecutive entries 
        represent the x and y components of one vertex."""
        for i in range(self.NV):
            des[2 * i] = source[i][0]
            des[2 * i + 1] = source[i][1]

    @ti.kernel
    def compute_b(
        self,
        f: ti.types.ndarray(),
        Kv: ti.types.ndarray(),
        h: ti.f32,
    ):
        # Now we write directly to self.b
        for i in range(2 * self.NV):
            self.b[i] = (f[i] + Kv[i] * h) * h


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

        # 4. Form system: A = M - h*D - h^2*K and b = h*(f + K*v)
        A = self.M - h * D - (h**2) * K

        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)

        # b = (force + h*K*vel)*h
        Kv = K @ self.vel_1D
        self.compute_b(self.b, self.force_1D, Kv, h)

        
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        dv = solver.solve(self.b)

        # 5. Update position & velocity
        self.updatePosVel(h, dv)

    @ti.kernel
    def spring2indices(self):
        for i in self.spring:
            """
            Note that in visiaulzation, we only need the direct springs. So the value of 
            indices is 2*NE, not 2*NE_total
            """
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
    


    @ti.kernel
    def compute_loss(self):
        # just a sample loss function
        self.loss[None] = 0.0
        for i in range(2 * self.NV):
            self.loss[None] += self.b[i] * self.b[i]



def main():
    
    ti.init(arch=ti.cpu)
    h = 0.02
    pause = False
    cloth = Cloth(N=10)
    with ti.ad.Tape(loss=cloth.loss):
        cloth.update(h)         # Forward simulation
        cloth.compute_loss()    # Evaluate the loss

    # Now Taichi has computed the gradients:
    # e.g. cloth.pos.grad, cloth.spring_ks.grad, etc.
    print("loss =", cloth.loss[None])
    print("d(loss)/d pos =", cloth.pos.grad.to_numpy()[:10])
    print("d(loss)/d spring_ks =", cloth.spring_ks.grad.to_numpy()[:10])

    """use_ggui = args.use_ggui
    if not use_ggui:
        gui = ti.GUI("Heterogeneous with bendering", res=(500, 500))
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
            window.show()"""
    return cloth

if __name__ == "__main__":
    cloth=main()
    print("Done")
