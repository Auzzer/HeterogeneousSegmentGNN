import taichi as ti
ti.init(arch=ti.gpu)  # or ti.init(arch=ti.cpu)

# Simulation parameters
n = 64                 # grid resolution in y and z directions
d = 8                  # number of layers (depth along x)
quad_size = 1.0 / n    # in-plane spacing (y-z plane)
thickness = 0.02       # spacing between layers (cloth thickness along x)
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
global_spring_Y = 1e2  # base spring stiffness, as requested
dashpot_damping = 1e3
drag_damping = 1

# 3D simulation state: now the grid indices are:
#   i for y (vertical), j for z (horizontal), and k for x (thickness)
x = ti.Vector.field(3, dtype=float, shape=(n, n, d))
v = ti.Vector.field(3, dtype=float, shape=(n, n, d))
# Heterogeneous spring stiffness field; each layer is modulated differently.
spring_Y_field = ti.field(dtype=float, shape=(n, n, d))

# For rendering we display one “representative” layer.
render_layer = d // 2
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = True

@ti.kernel
def initialize_mass_points():
    # Initialize the cloth so that its sheet is flat in the y-z plane.
    # The x coordinate comes from the layer index k (centered around 0),
    # y is vertical (with top row i==0 at y=0.6) and z is horizontal.
    for i, j, k in x:
        # x coordinate: spread along thickness, centered at 0.
        pos_x = thickness * (k - (d - 1) / 2)
        # y coordinate: i is vertical; top row (i==0) at y=0.6.
        pos_y = 0.6 - i * quad_size
        # z coordinate: j is horizontal, centered around 0.
        pos_z = j * quad_size - 0.5
        x[i, j, k] = ti.Vector([pos_x, pos_y, pos_z])
        v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def initialize_spring_field():
    # Assign each mass point a spring stiffness value.
    # In addition to a random factor in [1.0, 1.5],
    # add a modulation based on the layer index (k) so that different layers differ.
    for i, j, k in spring_Y_field:
        spring_Y_field[i, j, k] = global_spring_Y + 1000.0 * ti.random()

@ti.kernel
def initialize_mesh_indices():
    # Build mesh indices for the representative layer (render_layer) only.
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = i * (n - 1) + j
        # First triangle
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # Second triangle
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

@ti.kernel
def update_colors():
    # Update colors for the representative layer based on the local stiffness.
    # Map stiffness values in the range [global_spring_Y, 1.5*global_spring_Y] to t in [0,1],
    # then use t to interpolate from blue (low stiffness) to red (high stiffness).
    for i, j in ti.ndrange(n, n):
        t = (spring_Y_field[i, j, render_layer] - global_spring_Y) / (global_spring_Y * 0.5)
        t = ti.min(ti.max(t, 0.0), 1.0)
        colors[i * n + j] = ti.Vector([t, 0.0, 1.0 - t])

# Define spring offsets in 3D.
# We add in-plane connections (neighbors in the y-z plane) and direct cross-layer connections.
spring_offsets = []
if bending_springs:
    # In-plane neighbors: changes in i and j with no change in k.
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            spring_offsets.append(ti.Vector([di, dj, 0]))
    # Direct cross-layer connections (neighbors in the thickness direction).
    spring_offsets.append(ti.Vector([0, 0, 1]))
    spring_offsets.append(ti.Vector([0, 0, -1]))
else:
    for di in range(-2, 3):
        for dj in range(-2, 3):
            if di == 0 and dj == 0:
                continue
            if abs(di) + abs(dj) <= 2:
                spring_offsets.append(ti.Vector([di, dj, 0]))
    spring_offsets.append(ti.Vector([0, 0, 1]))
    spring_offsets.append(ti.Vector([0, 0, -1]))

@ti.kernel
def substep():
    # Apply gravity to all mass points.
    for I in ti.grouped(x):
        v[I] += gravity * dt

    # Compute spring forces.
    for I in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for offset in ti.static(spring_offsets):
            J = I + offset
            if 0 <= J[0] < n and 0 <= J[1] < n and 0 <= J[2] < d:
                x_diff = x[I] - x[J]
                v_diff = v[I] - v[J]
                d_normal = x_diff.normalized()
                current_dist = x_diff.norm()
                # Rest length: if the connection is cross-layer, use 'thickness';
                # otherwise, use the in-plane spacing.
                
                original_dist = thickness * abs(offset[2])
                if offset[2] == 0:
                    in_plane = ti.Vector([offset[0], offset[1]])
                    original_dist = quad_size * in_plane.norm()
                # Use the average stiffness of the two connected points.
                stiffness = (spring_Y_field[I] + spring_Y_field[J]) * 0.5
                force += -stiffness * d_normal * (current_dist / original_dist - 1)
                force += -v_diff.dot(d_normal) * d_normal * dashpot_damping * quad_size
        v[I] += force * dt

    # Apply drag damping and update positions.
    for I in ti.grouped(x):
        v[I] *= ti.exp(-drag_damping * dt)
        # Fix the top edge of the cloth (i == 0) for all layers.
        if I[0] == 0:
            v[I] = ti.Vector([0.0, 0.0, 0.0])
        else:
            x[I] += dt * v[I]

@ti.kernel
def update_vertices():
    # Use the representative layer for rendering.
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j, render_layer]

window = ti.ui.Window("Flat Cloth in y-z Plane", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

current_t = 0.0
initialize_mass_points()
initialize_spring_field()
initialize_mesh_indices()
update_colors()

while window.running:
    
    for i in range(substeps):
        substep()
        current_t += dt
    update_vertices()
    update_colors()

    # Set the camera so that we view along the x-axis,
    # so that the cloth (in the y-z plane) is clearly visible.
    camera.position(3.0, 0.0, 0.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)
    canvas.scene(scene)
    window.show()
