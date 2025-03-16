# HeterogeneousSegmentGNN

## Spring Mass Simulation

Consider the following illustration:

![Illustration](https://ics.uci.edu/~shz/courses/cs114/docs/proj3/images/fig1.jpg)

*Figure 1:* A cloth modeled as a grid of particles (nodes) connected by various types of springs (edges).

---
### 1. Variable Statement
#### 1.1 Particles (Nodes)

- **`pos`**: A 2D vector field holding the current positions of the cloth’s vertices (red dots). Dense field of size $NV = (N+1)^2$.

- **`initPos`**  Stores the initial (rest) positions of each vertex.  

- **`vel`**  The velocity of each particle (initialized to zero). 

- **`mass`**  The mass value at each vertex.  

#### 1.2 Edges (Springs)

Edges represent the forces between particles and are divided into three types:

##### a. Structural Springs (Stretch)
- **Description:** Connect directly adjacent vertices (horizontally and vertically) to resist stretching.
- **Variables:**
  - **`spring`** (indices `[0, NE)`): Each entry is a vector $[i,j]$ representing the two connected vertices.
  - **`rest_len`**: The rest length of each spring, computed from the initial positions:
    $$    L_{\text{rest}} = \| \text{pos}[i] - \text{pos}[j] \|$$
  - **`spring_ks`**: The stiffness for each structural/shear spring.
- **Storage:**  
  These variables are stored densely for each edge; the overall connectivity is later used to assemble sparse global matrices.

##### b. Shear Springs
- **Description:** Diagonal connections (e.g., from (i,j) to (i+1,j+1) or (i+1,j) to (i,j+1)) that resist in-plane shearing.
- **Variables:**  
  They are stored in the same index range as structural springs in the same variable `String` (starting at the end of the structural springs).

##### c. Bending (Flexion) Springs
- **Description:** Connect vertices that are two cells apart (horizontally or vertically) to model resistance to bending.
- **Variables:**
  - **`spring`** (indices `[NE, NE_total)`): Contains the connectivity for bending springs.
  - **`rest_len`**: Rest lengths for bending springs.
  - **`bend_ks`**: The bending stiffness for each bending spring, typically set as a fraction (e.g., 10%) of the average structural stiffness.
- **Storage:**  
  Dense for each edge; these edges are then assembled into sparse matrices along with the other springs.


#### 1.3 Global Vectors for the Sparse Solver

- **`vel_1D`**, **`force_1D`**, **`b`**  
  - **Description:** One-dimensional (flattened) arrays of length 2 * NV that store the velocity, force, and right-hand side for the global linear system.
  - **Purpose:** They are used for `ti.linalg.SparseSolver` which expects flat vectors.

---

#### 1.4 Global Sparse Matrices

- **Mass Matrix (M)**
  - **Description:** A diagonal matrix of size 2NV *2NV where each vertex contributes two degrees of freedom (x and y).
  - **Formula:**  
    $M_{ii} = m_i, \quad \text{for } i = 0, \dots, 2NV-1$


- **Stiffness Matrix ($K$)**
  - **Description:** Assembled from the local Jacobians $J_x$ (which depend on spring stiffness and position differences).


- **Damping Matrix ($D$)**
  - **Description:** Assembled from the local Jacobians $J_v$ (which capture the damping effect).


---

#### 1.5 Jacobian Fields

- **`Jx`**  
  - **Description:** For each edge, a $2 \times 2$ matrix representing:
    $J_x = \frac{\partial f(\text{edge},\,\text{pos})}{\partial \text{pos}}$,
    where     $J_x = \left(I - \frac{L_{\text{rest}}}{l}\left(I - \frac{\Delta x\,\Delta x^T}{l^2}\right)\right) k$, with $I$ as the identity matrix, $l = \|\Delta x\|$, and $k$ the edge’s stiffness.
  - **Storage:** Dense Matrix.field of size $NE\_total$.

- **`Jv`**  
  - **Description:** For each edge, a $2 \times 2$ matrix representing:
    $J_v = \frac{\partial f(\text{edge},\,\text{vel})}{\partial \text{vel}},$
    often set to:
    $J_v = k_d I,$
    where $k_d$ is the damping constant.
  - **Storage:** same as Jx

- **`Jf`**  
  - **Description:** The Hessian for fixed vertices used in fixed-point constraints, typically:
    $J_f = -k_f I$.
  - **Storage:** Dense, for each fixed vertex.



### 2. Forme the Whole System and Simulation Update
The force exerted by each spring depends on the edge's properties(i.e. the stiffness) and the positions of the two connected vertices. We denote this as: $f(\text{edge, pos})$. Note that for each edge it has two character: length and the stiffness.

#### 2.1 Force Function $f$ (edge, pos)

For a given spring (edge) connecting vertices $i$ and $j$, the force is computed by:

$$
\mathbf{f}=k\left(\|\Delta \mathbf{x}\|-L_{\text {rest }}\right) \frac{\Delta \mathbf{x}}{\|\Delta \mathbf{x}\|}
$$

where:
- $\Delta \mathbf{x}=\text{pos}[i]-\text{pos}[j]$
- $L_{\text {rest }}$ is the rest length (stored in rest_len for that edge)
- $k$ is the spring stiffness-either taken from spring_ks (for structural/shear springs) or bend_ks (for bending springs)

Thus, we write this as:

$$
f(\text { edge, pos }) \equiv k(\text { edge })\left(\|\Delta \mathbf{x}\|-L_{\text {rest }}(\text { edge })\right) \frac{\Delta \mathbf{x}}{\|\Delta \mathbf{x}\|}
$$

#### 2.2 Linearization and Jacobians

Because the force is nonlinear in pos, the SIG98 method linearizes it around the current state. This yields two key Jacobians:
- Position Jacobian $J_x$ :

This matrix represents the derivative of the force with respect to positions:

$$
J_x(\text { edge }, \text { pos })=\frac{\partial f(\text { edge, pos })}{\partial \text { pos }}
$$


In our 2D case, $J_x$ is a $2 \times 2$ matrix for each edge. The formula used is:

$$
J_x=\left(I-\frac{L_{\text {rest }}}{l}\left(I-\frac{\Delta x \Delta x^T}{l^2}\right)\right) \cdot k
$$

where:
- $I$ is the $2 \times 2$ identity matrix,
- $l=\|\Delta x\|$ (with $\Delta x=\text{pos}[i]-\text{pos}[j]$ ),
- $k$ is the stiffness for that particular edge.

In our code, this is implemented in compute_Jacobians as:


> self.Jx[i] = (I - self.rest_len[i] * inv_l * (I - dxtdx * (inv_l**2))) * k_current

where k_current is selected based on whether the edge is structural/shear or bending.

- Velocity Jacobian $J_v$ 
  
This is the derivative of the damping force with respect to the velocity：
$J_v(\text { edge }, \text { vel })=\frac{\partial f_{\text {damp }}}{\partial \mathrm{vel}}$


For a simple damping model $f_{\text {damp }}=-k_d\left(\text{vel}_i-\text{vel}_j\right)$ ，we have：

$J_v=k_d I$

with $I$ again being the $2 \times 2$ identity．In our code，we simply set：
> self.Jv[i] = self.kd * I


#### 2.3 Assembling Global Matrix
Using these local Jacobians，we build the global system．The key equation from the paper is：

$$
\mathbf{A}=\mathbf{M}-h \frac{\partial \mathbf{f}}{\partial \mathrm{vel}}-h^2 \frac{\partial \mathbf{f}}{\partial \mathrm{pos}}
$$


- $\mathbf{M}$ is the mass matrix（built from mass，a diagonal sparse matrix of size $2 N V \times 2 N V$ ）．
- $D$ is the global damping sparse matrix，assembled from the $J_v(edge，vel)$ contributions：$D := \frac{\partial \mathbf{f}}{\partial \mathrm{vel}}$


- $K$ is the Sparse stiffness matrix，assembled from the $J_x$（edge，pos）contributions：$K := \frac{\partial \mathbf{f}}{\partial \mathrm{pos}}$

The assembly is done by iterating over each edge（spring）and adding its $2 \times 2$ block contributions to the appropriate locations in $K$ and $D$. For example，for an edge connecting vertices $i$ and $j$, the contribution to $K$ is:
$$
\left[\begin{array}{cc}
-J_x & J_x \\
J_x & -J_x
\end{array}\right]
$$

This is because by Newton's third law, the force on vertex $i$ is$\mathbf{f}_i=f\left(\mathbf{x}_i, \mathbf{x}_j\right)$
and the force on vertex $j$ is$\mathbf{f}_j=-f\left(\mathbf{x}_i, \mathbf{x}_j\right)$.
When we linearize the forces around the current state, we need to compute the Jacobian, i.e., the derivatives of these forces with respect to the positions $\mathbf{x}_i$ and $\mathbf{x}_j$.

Let $J_x=\frac{\partial f}{\partial \mathbf{x}}$. Because the force is equal and opposite, the derivative of the force on $i$ with respect to $\mathbf{x}_i$ is $-J_x$, and the derivative with respect to $\mathbf{x}_j$ is $+J_x$. Similarly, for the force on $j$ the roles are reversed. When we write these contributions in block matrix form, we get

$$
\left(\begin{array}{ll}
\frac{\partial \mathbf{f}_i}{\partial \mathbf{x}_i} & \frac{\partial \mathbf{f}_i}{\partial \mathbf{x}_j} \\
\frac{\partial \mathbf{f}_j}{\partial \mathbf{x}_i} & \frac{\partial \mathbf{f}_j}{\partial \mathbf{x}_j}
\end{array}\right)=\left(\begin{array}{cc}
-J_x & J_x \\
J_x & -J_x
\end{array}\right) .
$$


So for each spring connecting vertices $i$ and $j$, the contribution to the global stiffness matrix $K$ is of the form:

$$
\left[\begin{array}{cc}
-J_x & J_x \\
J_x & -J_x
\end{array}\right]
$$

>K[2 * idx1 + m, 2 * idx1 + n] -= self.Jx[i][m, n]\
>K[2 * idx1 + m, 2 * idx2 + n] += self.Jx[i][m, n]\
>K[2 * idx2 + m, 2 * idx1 + n] += self.Jx[i][m, n]\
>K[2 * idx2 + m, 2 * idx2 + n] -= self.Jx[i][m, n]

And a similar pattern is used for $D$ with $J_v$. 

#### 2.4 Global System for Simulation Update
The final system solved is：$A \Delta v=b$, with:
- $A=M-h D-h^2 K$（all assembled as sparse matrices），
- $b=h(f+K v)$（where $f$ is the current force computed from $f(edge，pos)$.\
This system is solved to update the velocities，and then the positions are updated：
$$
v_{n+1}=v_n+\Delta v, \quad x_{n+1}=x_n+h v_{n+1}
$$

### BP to torch



### GNN modeling towards the Spring Mass System

## Folder Struture

### Simulattion
**Note**: runs on mac may trigger the problem like:
https://discussions.apple.com/thread/255761734?sortBy=rank.
It's caused by: https://github.com/homebrew-zathura/homebrew-zathura/issues/129#issuecomment-2360382975. It doesn't affect the simulation, you can ignore it. 

#### 1. original
adopted from: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/implicit_mass_spring.py

#### 2. heterogeneous.py
make the spring heterogeneous

#### 3. HeteroWithBending2.py

Questions
1. lung mesh? image to mesh models:
2. 