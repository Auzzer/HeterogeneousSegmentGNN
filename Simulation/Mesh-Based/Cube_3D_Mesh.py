import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# Simulation parameters
n = 30             # number of segments in one dimension
dx = 1 / n         # spacing between particles
dt = 1e-4          # time step
gravity = ti.Vector([0, -9.8, 0])
num_particles = (n + 1) * (n + 1)
num_triangles = 2 * n * n  # two triangles per grid cell

# Fields for particle positions and velocities
pos = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
# Field for triangle connectivity (each triangle is represented by 3 particle indices)
triangles = ti.field(dtype=ti.i32, shape=(num_triangles, 3))

@ti.kernel
def initialize():
    # Initialize particle positions on a flat grid and zero initial velocities.
    for i, j in ti.ndrange(n + 1, n + 1):
        idx = i * (n + 1) + j
        # Place particles with y = 1.0 for an initial elevation.
        pos[idx] = ti.Vector([i * dx, 1.0, j * dx])
        vel[idx] = ti.Vector([0, 0, 0])
        
    # Build triangle connectivity for each grid cell.
    tri = 0
    for i, j in ti.ndrange(n, n):
        idx = i * (n + 1) + j
        idx_right = (i + 1) * (n + 1) + j
        idx_down = i * (n + 1) + (j + 1)
        idx_diag = (i + 1) * (n + 1) + (j + 1)
        # First triangle in the cell
        triangles[tri, 0] = idx
        triangles[tri, 1] = idx_right
        triangles[tri, 2] = idx_down
        tri += 1
        # Second triangle in the cell
        triangles[tri, 0] = idx_right
        triangles[tri, 1] = idx_diag
        triangles[tri, 2] = idx_down
        tri += 1

@ti.kernel
def substep():
    # Simple explicit time integration (Euler method) with gravity.
    for i in range(num_particles):
        # Apply gravity to velocity
        vel[i] += gravity * dt
        # Update position using the current velocity
        pos[i] += vel[i] * dt
        # Simple ground collision: reset if below y = 0
        if pos[i][1] < 0:
            pos[i][1] = 0
            vel[i][1] = 0

def main():
    initialize()
    gui = ti.GUI("Mesh Simulation with Taichi", (800, 800))
    while gui.running:
        # Run multiple simulation substeps per frame for stability.
        for _ in range(50):
            substep()
        # Visualize the mesh: draw lines for each triangle's edges.
        for t in range(num_triangles):
            i0 = triangles[t, 0]
            i1 = triangles[t, 1]
            i2 = triangles[t, 2]
            p0 = pos[i0].to_numpy() * 500  # scaling for visualization
            p1 = pos[i1].to_numpy() * 500
            p2 = pos[i2].to_numpy() * 500
            gui.line(p0, p1, radius=1, color=0x0)
            gui.line(p1, p2, radius=1, color=0x0)
            gui.line(p2, p0, radius=1, color=0x0)
        gui.show()

if __name__ == '__main__':
    main()
