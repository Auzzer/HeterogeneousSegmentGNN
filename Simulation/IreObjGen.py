import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # just to enable 3D projection

def build_3d_shape(N=8, p_keep_bottom=0.3):
    """
    Build a (N+1)x(N+1)x(N+1) grid of points.
    - For the bottom layer j=0, keep each point with probability p_keep_bottom.
    - For layers j=1..N: if there's NO particle below, decide randomly whether to keep this one.
      If there IS a particle below, automatically keep this one.
    Returns:
       active_3d: a bool array of shape (N+1, N+1, N+1) 
    """
    active_3d = np.zeros((N+1, N+1, N+1), dtype=bool)

    # 1) Bottom layer random
    for i in range(N+1):
        for k in range(N+1):
            if np.random.rand() < p_keep_bottom:
                active_3d[i,0,k] = True
            else:
                active_3d[i,0,k] = False

    # 2) Layer by layer
    for j in range(1, N+1):
        for i in range(N+1):
            for k in range(N+1):
                if active_3d[i,j-1,k]:
                    # If the particle below is active, keep this one
                    active_3d[i,j,k] = True
                else:
                    # If below is inactive, randomly decide
                    if np.random.rand() < 0.3:
                        active_3d[i,j,k] = True
                    else:
                        active_3d[i,j,k] = False
    return active_3d

def show_3d_shape(active_3d):
    """
    Plot all active points in 3D using matplotlib.
    """
    N = active_3d.shape[0] - 1

    # We'll map i->x, j->y, k->z in some scaled region
    x_list, y_list, z_list = [], [], []
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                if active_3d[i,j,k]:
                    # example scaling
                    x = 0.25 + 0.5*(i/N)
                    y = 0.75 + 0.5*(j/N)
                    z = 0.25 + 0.5*(k/N)
                    x_list.append(x)
                    y_list.append(y)
                    z_list.append(z)

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    z_arr = np.array(z_list)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_arr, y_arr, z_arr, c='blue', marker='o', s=20)

    # A little cosmetic setup
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Randomly Dropped 3D Shape')
    plt.show()

# Example usage:
if __name__ == "__main__":
    N = 8
    p_keep_bottom = 0.4

    active_3d = build_3d_shape(N, p_keep_bottom)
    show_3d_shape(active_3d)
