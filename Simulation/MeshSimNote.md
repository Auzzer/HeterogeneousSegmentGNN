# 1. Liu's Quasi-Newton Methods for Real-Time Simulation of Hyperelastic Materials

## Motivation
- Real-time Physics Simulation Requirements
In computer graphics, to achieve real-time or near-real-time physics simulation in applications such as games and interactive systems (e.g., surgical simulators), researchers require fast and stable numerical methods.
Traditionally, there are two popular approaches:
1. Position Based Dynamics (PBD)
2. Projective Dynamics (PD)

Both methods are very fast and have been widely applied in industry; however, the range of elastic materials they support is relatively limited, making it challenging to handle more complex nonlinear materials (such as classical continuum mechanics models like Neo-Hookean, Mooney-Rivlin, etc.).

- More General Material Models
For example, models such as Neo-Hookean or Spline-based (a “spline-based material” proposed by Xu et al. in 2015) allow artists and technical animators to more conveniently adjust material properties to achieve the desired visual or animation effects.

When these nonlinear materials are solved using the classical Newton’s method, it typically requires one or more iterations per frame—each iteration involves solving a linear system based on a changing Hessian (second derivative matrix), resulting in a significant computational cost that makes real-time performance difficult to attain.

- Existing Methods and Challenges
1. In Newton’s method, the Hessian changes over iterations, requiring constant updates and new linear system solves. Moreover, the Hessian may be indefinite, necessitating “definiteness fixes” to ensure positive definiteness, which further increases computational overhead.
2. Even performing just one iteration of Newton’s method (often referred to as “one-iteration Newton’s method”) is, in many cases, too slow or not sufficiently accurate.

## Material setting

# 2. Liu's Fast Simulation of Mass-Spring Systems
A note of equation (12)
Given three particles with 2 springs:
- Spring 1: Connects particles 1 and 2 , stiffness $k_1$, direction $\mathbf{d}_1 \in \mathbb{R}^3$.
- Spring 2: Connects particles 2 and 3, stiffness $k_2$, direction $\mathbf{d}_2 \in \mathbb{R}^3$.
  
### Coordinate and Direction Vectors
Particle coordinates (stacked in $\mathbf{x} \in \mathbb{R}^9$ ):

$\mathbf{x}=[p_1, p_2, p_3]^T, where \quad \mathbf{p}_i \in \mathbb{R}^3$ 
and Spring directions (stacked in $\mathbf{d} \in \mathbb{R}^6$ ):

$
\mathbf{d}=[d_1, d_2]^T, \quad \mathbf{d}_i \in \mathbb{R}^3
$

For the  Incidence Vectors $\left(\mathbf{A}_i\right)$ and Indicator Vectors $\left(\mathbf{S}_i\right)$

Spring 1 (connects particles 1 and 2):
Incidence vector $\mathbf{A}_1 \in \mathbb{R}^3$ :
$
\mathbf{A}_1=\left[\begin{array}{c}
1 \\
-1 \\
0
\end{array}\right].
$
And Indicator vector $\mathbf{S}_1 \in \mathbb{R}^2$ :
$
\mathbf{S}_1=\left[\begin{array}{l}
1 \\
0
\end{array}\right]
$


Spring 2 (connects particles 2 and 3):
Incidence vector $\mathbf{A}_2 \in \mathbb{R}^3$ :
$
\mathbf{A}_2=\left[\begin{array}{c}
0 \\
1 \\
-1
\end{array}\right]
$
And indicator vector $\mathrm{S}_2 \in \mathbb{R}^2$:
$
\mathbf{S}_2=\left[\begin{array}{l}
0 \\
1
\end{array}\right]
$

### THen Constructing the Laplacian Matrix L

Step 1: Compute $\mathbf{A}_i \mathbf{A}_i^{\top}$ for Each Spring
- Spring 1:

$$
\mathbf{A}_1 \mathbf{A}_1^{\top}=\left[\begin{array}{ccc}
1 & -1 & 0 \\
-1 & 1 & 0 \\
0 & 0 & 0
\end{array}\right]
$$

- Spring 2:

$$
\mathbf{A}_2 \mathbf{A}_2^{\top}=\left[\begin{array}{ccc}
0 & 0 & 0 \\
0 & 1 & -1 \\
0 & -1 & 1
\end{array}\right]
$$


Step 2: Sum Contributions with Stiffness Scaling

$$
\sum_{i=1}^2 k_i \mathbf{A}_i \mathbf{A}_i^{\top}=k_1 \mathbf{A}_1 \mathbf{A}_1^{\top}+k_2 \mathbf{A}_2 \mathbf{A}_2^{\top}=\left[\begin{array}{ccc}
k_1 & -k_1 & 0 \\
-k_1 & k_1+k_2 & -k_2 \\
0 & -k_2 & k_2
\end{array}\right]
$$

Step 2: Sum Contributions with Stiffness Scaling

$$
\sum_{i=1}^2 k_i \mathbf{A}_i \mathbf{A}_i^{\top}=k_1 \mathbf{A}_1 \mathbf{A}_1^{\top}+k_2 \mathbf{A}_2 \mathbf{A}_2^{\top}=\left[\begin{array}{ccc}
k_1 & -k_1 & 0 \\
-k_1 & k_1+k_2 & -k_2 \\
0 & -k_2 & k_2
\end{array}\right]
$$


Step 3: Kronecker Product with $\mathbf{I}_3$

$$
\mathbf{L}=\left(\sum_{i=1}^2 k_i \mathbf{A}_i \mathbf{A}_i^{\top}\right) \otimes \mathbf{I}_3 \in \mathbb{R}^{9 \times 9}
$$


Explicit Block Structure:

$$
\mathbf{L}=\left[\begin{array}{ccc}
k_1 \mathbf{I}_3 & -k_1 \mathbf{I}_3 & \mathbf{0}_3 \\
-k_1 \mathbf{I}_3 & \left(k_1+k_2\right) \mathbf{I}_3 & -k_2 \mathbf{I}_3 \\
\mathbf{0}_3 & -k_2 \mathbf{I}_3 & k_2 \mathbf{I}_3
\end{array}\right]
$$

- Diagonal Blocks: Total stiffness connected to each particle.
- Off-Diagonal Blocks: Negative stiffness for connected particle pairs.

### Constructing the Coupling Matrix J

Step 1: Compute $\mathbf{A}_i \mathbf{S}_i^{\top}$ for Each Spring
- Spring 1:

$$
\mathbf{A}_1 \mathbf{S}_1^{\top}=\left[\begin{array}{cc}
1 & 0 \\
-1 & 0 \\
0 & 0
\end{array}\right]
$$

- Spring 2:

$$
\mathbf{A}_2 \mathbf{S}_2^{\top}=\left[\begin{array}{cc}
0 & 0 \\
0 & 1 \\
0 & -1
\end{array}\right]
$$


Step 2: Sum Contributions with Stiffness Scaling

$$
\sum_{i=1}^2 k_i \mathbf{A}_i \mathbf{S}_i^{\top}=k_1 \mathbf{A}_1 \mathbf{S}_1^{\top}+k_2 \mathbf{A}_2 \mathbf{S}_2^{\top}=\left[\begin{array}{cc}
k_1 & 0 \\
-k_1 & k_2 \\
0 & -k_2
\end{array}\right]
$$


Step 3: Kronecker Product with $\mathbf{I}_3$

$$
\mathbf{J}=\left(\sum_{i=1}^2 k_i \mathbf{A}_i \mathbf{S}_i^{\top}\right) \otimes \mathbf{I}_3 \in \mathbb{R}^{9 \times 6}
$$


Explicit Block Structure:

$$
\mathbf{J}=\left[\begin{array}{cc}
k_1 \mathbf{I}_3 & \mathbf{0}_3 \\
-k_1 \mathbf{I}_3 & k_2 \mathbf{I}_3 \\
\mathbf{0}_3 & -k_2 \mathbf{I}_3
\end{array}\right]
$$

- Column Blocks: Each corresponds to a spring direction $\mathbf{d}_i$, affecting connected particles.


### Physical Interpretation
Finally, follow the original paper, we have:
- L encodes the stiffness interactions between particles. Diagonal blocks represent "self-stiffness," while off-diagonal blocks represent "coupling stiffness" between connected particles.
- J maps the spring directions $\mathbf{d}_i$ to the particle displacements. Each column block ensures that the direction of a spring influences only its connected particles.

### Example Energy Calculation

The total spring potential energy is:

$$
\frac{1}{2} \mathbf{x}^{\top} \mathbf{L} \mathbf{x}-\mathbf{x}^{\top} \mathbf{J d} .
$$

- Quadratic Term ( $\mathbf{x}^{\top} \mathbf{L} \mathbf{x}$ ): energy from particle displacements relative to each other.
- Linear Term ( $-\mathbf{x}^{\top} \mathbf{J d}$ ):  particle positions with spring rest directions.