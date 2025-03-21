import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_layered_shape_shrink(N=8, p_shrink=0.3):
    """
    Generate a layered solid shape on an (N+1)x(N+1)x(N+1) grid with:
      1. A randomly picked starting layer j0 (with 0 < j0 < N) that is forced to be an 8×8 block.
      2. For upward layers (j0+1 to N) the boundaries shrink in both i and k directions 
         (with probability p_shrink) so that the region becomes smaller.
      3. For downward layers (j0-1 down to 0), the same shrink rule is applied.
    
    For a grid of size 9×9 per layer (i.e. N=8), the starting region is indices 1 through 8.
    
    Returns:
      active_3d: a boolean array of shape (N+1, N+1, N+1) with True for active points.
    """
    grid_size = N + 1
    active_3d = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    # 1. Randomly pick a starting layer j0 (avoid j=0 and j=N)
    j0 = np.random.randint(1, N)
    print(f"Chosen starting layer: {j0}")
    
    # 2. Define the starting layer's region as exactly an 8×8 block.
    # For a grid of size 9, use indices 1 through 8.
    i_min0, i_max0 = 1, grid_size   # active i indices 1 ... 8
    k_min0, k_max0 = 1, grid_size   # active k indices 1 ... 8
    
    # Fill layer j0 completely in that 8×8 region.
    for i in range(i_min0, i_max0):
        for k in range(k_min0, k_max0):
            active_3d[i, j0, k] = True
            
    # Helper function to adjust boundaries for shrinking.
    def adjust_boundaries(i_min, i_max, k_min, k_max):
        new_i_min = i_min + (1 if np.random.rand() < p_shrink else 0)
        new_i_max = i_max - (1 if np.random.rand() < p_shrink else 0)
        new_k_min = k_min + (1 if np.random.rand() < p_shrink else 0)
        new_k_max = k_max - (1 if np.random.rand() < p_shrink else 0)
        # Ensure the region remains at least 2×2.
        if new_i_max - new_i_min < 2:
            new_i_min, new_i_max = i_min, i_max
        if new_k_max - new_k_min < 2:
            new_k_min, new_k_max = k_min, k_max
        return new_i_min, new_i_max, new_k_min, new_k_max
    
    # 3. For upward layers (j0+1 to grid_size-1), shrink boundaries.
    curr_i_min, curr_i_max = i_min0, i_max0
    curr_k_min, curr_k_max = k_min0, k_max0
    for j in range(j0+1, grid_size):
        curr_i_min, curr_i_max, curr_k_min, curr_k_max = adjust_boundaries(
            curr_i_min, curr_i_max, curr_k_min, curr_k_max
        )
        for i in range(curr_i_min, curr_i_max):
            for k in range(curr_k_min, curr_k_max):
                active_3d[i, j, k] = True
                
    # 4. For downward layers (j0-1 down to 0), also shrink boundaries.
    curr_i_min, curr_i_max = i_min0, i_max0
    curr_k_min, curr_k_max = k_min0, k_max0
    for j in range(j0-1, -1, -1):
        curr_i_min, curr_i_max, curr_k_min, curr_k_max = adjust_boundaries(
            curr_i_min, curr_i_max, curr_k_min, curr_k_max
        )
        for i in range(curr_i_min, curr_i_max):
            for k in range(curr_k_min, curr_k_max):
                active_3d[i, j, k] = True
                
    return active_3d

def plot_3d(active_3d):
    """
    Plot the active points in 3D.
    Grid indices (i, j, k) are mapped to coordinates:
      x = 0.25 + 0.5*(i/N),  y = 0.75 + 0.5*(j/N),  z = 0.25 + 0.5*(k/N)
    Axis limits are explicitly set.
    """
    grid_size = active_3d.shape[0]
    N = grid_size - 1
    x_list, y_list, z_list = [], [], []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if active_3d[i, j, k]:
                    x = 0.25 + 0.5*(i/N)
                    y = 0.75 + 0.5*(j/N)
                    z = 0.25 + 0.5*(k/N)
                    x_list.append(x)
                    y_list.append(y)
                    z_list.append(z)
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_list, y_list, z_list, s=30, c='blue', alpha=0.7)
    
    # Set explicit axis limits.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)  # y covers 0.75 to 1.25 with margin
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Layered Solid Shape with 8×8 Starting Layer and Upward & Downward Shrink')
    plt.show()

if __name__ == "__main__":
    active = generate_layered_shape_shrink(N=8, p_shrink=0.3)
    plot_3d(active)
