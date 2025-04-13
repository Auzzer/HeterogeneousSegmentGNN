import numpy as np
import meshio

def tetra_volume(p0, p1, p2, p3):
    """Compute the volume of a tetrahedron given its four vertices."""
    mat = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
    return np.abs(np.linalg.det(mat)) / 6.0

def process_mesh(file_name):
    """
    Process a tetrahedral mesh file to extract:
      - Vertex positions (points)
      - Unique edges (as pairs of vertex indices)
      - Rest lengths for each edge
      - Effective edge stiffness (volume weighted from tetra stiffness)
      
    Assumes the mesh has tetrahedral cells stored under the key "tetra" and
    a per-tetra cell_data entry "c_labels" that gives a material label.
    """
    # 1. Load the mesh
    mesh = meshio.read(file_name)
    points = mesh.points  # shape (N_points, 3)
    # Get tetrahedral cell connectivity
    tetra = mesh.cells_dict["tetra"]  # shape (N_tets, 4)
    # Get cell data for tetrahedra.
    tetra_labels = mesh.cell_data["c_labels"][0]  # shape (N_tets,)


    # 2. Map material labels to stiffness values.
    material_stiffness_map = {
        1: 1.0e5,
        2: 2.0e5,
        3: 3.0e5,   
        4: 4.2e5,
        5: 5.5e5,
        6: 6.8e5,
        7: 7.0e5,
        8: 8.5e5,
        9: 9.8e5,
        10: 1.0e6,
        11: 2.3e6
    }
    
    N_tets = tetra.shape[0]
    tetra_volumes = np.zeros(N_tets, dtype=np.float32)
    tetra_stiffness = np.zeros(N_tets, dtype=np.float32)
    
    # Compute volume and assign stiffness for each tetrahedron.
    for i in range(N_tets):
        inds = tetra[i]
        p0, p1, p2, p3 = points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]
        vol = tetra_volume(p0, p1, p2, p3)
        tetra_volumes[i] = vol
        label = tetra_labels[i]
        tetra_stiffness[i] = material_stiffness_map.get(label, 1.0e5)  # default value if label not found

    # 3. Extract unique edges from tetrahedra.
    # Each tetrahedron has 6 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edge_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    edge_data = {}  #  a list with key: (min_vertex, max_vertex), value: list of (volume, tetra stiffness)
    
    for t in range(N_tets):
        inds = tetra[t]
        vol = tetra_volumes[t]
        ks = tetra_stiffness[t]
        for (a, b) in edge_pairs:
            vA, vB = inds[a], inds[b]
            # Store edge as (min, max) to ensure uniqueness
            edge = tuple(sorted((vA, vB)))
            if edge not in edge_data:
                edge_data[edge] = []
            edge_data[edge].append((vol, ks))
    
    # 4. For each unique edge, compute effective stiffness (volume weighted) and rest length.
    edges_list = []
    edge_stiffness_list = []
    edge_rest_length_list = []
    for edge, contributions in edge_data.items():
        total_volume = sum(vol for vol, _ in contributions)
        weighted_stiffness = sum(vol * ks for vol, ks in contributions)
        effective_stiffness = weighted_stiffness / total_volume if total_volume > 0 else 0.0
        edges_list.append(edge)
        edge_stiffness_list.append(effective_stiffness)
        # Compute rest length as the Euclidean distance between the two vertices.
        vA, vB = edge
        rest_length = np.linalg.norm(points[vA] - points[vB])
        edge_rest_length_list.append(rest_length)
    
    # Convert lists to numpy arrays.
    edges_array = np.array(edges_list, dtype=np.int32)  # shape (N_edges, 2)
    edge_stiffness_array = np.array(edge_stiffness_list, dtype=np.float32)  # shape (N_edges,)
    edge_rest_length_array = np.array(edge_rest_length_list, dtype=np.float32)  # shape (N_edges,)
    
    return points, edges_array, edge_rest_length_array, edge_stiffness_array

if __name__ == "__main__":
    
    file_name = "Simulation/mesh_files/case1_T00_lung_regions_11.xdmf"
    points, edges, rest_lengths, edge_stiffness = process_mesh(file_name)
    
    print("Number of vertices:", points.shape[0])
    print("Number of unique edges:", edges.shape[0])
    print("First 5 edges and their properties:")
    for i in range(min(5, edges.shape[0])):
        print(f"Edge {edges[i]}: Rest Length = {rest_lengths[i]:.4f}, Stiffness = {edge_stiffness[i]:.2e}")
    """
    Number of vertices: 9471
    Number of unique edges: 60736"""