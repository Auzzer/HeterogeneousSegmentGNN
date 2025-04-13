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
## Background and Problem Defination
one page
1. Setting: Discrete Time Integration

We have $m$ particles in $\mathbb{R}^3$ (so $3 m$ coordinates in total). Time is discretized into steps of size $h$, so the system's state (the positon) at step $n$ is $\mathbf{q}_n \in \mathbb{R}^{3 m}$. The forces are given by a function

$$
\mathbf{f}(\mathbf{q})=-\nabla E(\mathbf{q})
$$

where $E$ is a (generally non-linear, possibly non-convex) potential energy, and $\mathbf{M}$ is a (diagonal) mass matrix.

2. Implicit Euler Update Equations

Implicit Euler for a mechanical system can be written as:
1. Position update

$$
\mathbf{q}_{n+1}=\mathbf{q}_n+h \mathbf{v}_{n+1}
$$

2. Velocity update

$$
\mathbf{v}_{n+1}=\mathbf{v}_n+h \mathbf{M}^{-1} \mathbf{f}\left(\mathbf{q}_{n+1}\right)
$$


In words, the velocity at the next time step depends on the forces $\mathbf{f}\left(\mathbf{q}_{n+1}\right)$ evaluated at the new position $\mathbf{q}_{n+1}$. This "backward look" (using $\mathbf{q}_{n+1}$ ) is why it's called implicit Euler.
3. Eliminating Velocities to Get a Single Equation in $\mathbf{q}$

Because we know

$$
h \mathbf{v}_n=\mathbf{q}_n-\mathbf{q}_{n-1}, \quad h \mathbf{v}_{n+1}=\mathbf{q}_{n+1}-\mathbf{q}_n
$$

we can substitute these into the velocity update. After rearranging:

$$
\mathbf{q}_{n+1}-2 \mathbf{q}_n+\mathbf{q}_{n-1}=h^2 \mathbf{M}^{-1} \mathbf{f}\left(\mathbf{q}_{n+1}\right) (5)
$$


we can see this is a discrete version of Newton's $\mathbf{F}=\mathbf{M} \mathbf{a}$, except a (acceleration) is approximated by the finite-difference $\frac{\mathbf{q}_{n+1}-2 \mathbf{q}_n+\mathbf{q}_{n-1}}{h^2}$.


Equation (5) is a nonlinear system in $\mathbf{q}_{n+1}$ because $\mathbf{f}\left(\mathbf{q}_{n+1}\right)$ is nonlinear. We would typically solve it with a Newton (or quasi-Newton) method at each timestep.Recall the classical approach (Baraff and Witkin, 1998) linearizes the force around the known state $\mathbf{q}_n$, i.e.,

$$
\mathbf{f}\left(\mathbf{q}_{n+1}\right) \approx \mathbf{f}\left(\mathbf{q}_n\right)+\left(\left.\nabla \mathbf{f}\right|_{\mathbf{q}_n}\right)\left(\mathbf{q}_{n+1}-\mathbf{q}_n\right)
$$

and then solves the resulting linear system.

new slide

Reformulating as an Optimization Problem

A key insight is that solving equation (5) is equivalent to finding a minimum (a critical point) of a certain function. To see it, define:
- $\mathbf{x}:=\mathbf{q}_{n+1}$ (the unknown new position),
- $\mathbf{y}:=2 \mathbf{q}_n-\mathbf{q}_{n-1}$ (a known combination of old positions).

Equation (5) becomes:

$$
\mathbf{M}(\mathbf{x}-\mathbf{y})=h^2 \mathbf{f}(\mathbf{x})(7)
$$


Because $\mathrm{f}(\mathbf{x})=-\nabla E(\mathbf{x})$, this rearranges to

$$
\mathbf{M}(\mathbf{x}-\mathbf{y})+h^2 \nabla E(\mathbf{x})=0
$$


Notice that this is exactly the gradient of the function

$$
g(\mathbf{x})=\frac{1}{2}(\mathbf{x}-\mathbf{y})^T \mathbf{M}(\mathbf{x}-\mathbf{y})+h^2 E(\mathbf{x}) (8)
$$

with respect to $\mathbf{x}$. 

$$
\nabla g(\mathbf{x})=\mathbf{M}(\mathbf{x}-\mathbf{y})+h^2 \nabla E(\mathbf{x})
$$


Hence, setting $\nabla g(\mathbf{x})=0$ is equivalent to solving $\mathbf{M}(\mathbf{x}-\mathbf{y})=h^2 \mathbf{f}(\mathbf{x})$.
Therefore:
Implicit Euler is the same as solving the optimization problem

$$
\min _{\mathbf{x}} \underbrace{\frac{1}{2}(\mathbf{x}-\mathbf{y})^T \mathbf{M}(\mathbf{x}-\mathbf{y})+h^2 E(\mathbf{x})}_{=: g(\mathbf{x})}
$$


This viewpoint is sometimes called variational implicit Euler or optimization implicit Euler. In practice, we still need Newton or a similar method to minimize $g(\mathbf{x})$.

Connection to Position-Based Dynamics (PBD)

Position-Based Dynamics [Müller et al. 2007] is often described as a "constraint projection" method rather than a force-based method. However, if we define a potential energy $E_{P B D}$ whose terms are the squared constraint violations, then PBD is effectively trying to reduce

$$
g(\mathbf{x})=\frac{1}{2}(\mathbf{x}-\mathbf{y})^T \mathbf{M}(\mathbf{x}-\mathbf{y})+h^2 E_{P B D}(\mathbf{x})
$$

in a Gauss-Seidel-like fashion (i.e., iteratively projecting constraints one by one). Two key differences from the "full" implicit Euler approach are:
1. PBD typically ignores the inertia term explicitly (i.e., $(\mathbf{x}-\mathbf{y})^T \mathbf{M}(\mathbf{x}-\mathbf{y})$ ) during the local constraint projection.
2. Each "constraint projection" step does not account for how it might affect the other constraints" terms in $E_{P B D}$.

Despite being a heuristic, PBD often manages to do a decent job of reducing that overall energy.

"new page"
## Main part
1. Hooke's Law and the Reformulation

Hooke's Law (Equation 9)
A single spring connecting points $\mathbf{p}_1$ and $\mathbf{p}_2 \in \mathbb{R}^3$ with rest length $r$ has the potential:

$$
\frac{1}{2} k\left(\left\|\mathbf{p}_1-\mathbf{p}_2\right\|-r\right)^2
$$

where $k$ is the spring stiffness.
Lemma: Reformulating the Spring Potential
They introduce an auxiliary vector $\mathbf{d} \in \mathbb{R}^3$ subject to $\|\mathbf{d}\|=r$. Then

$$
\left(\left\|\mathbf{p}_1-\mathbf{p}_2\right\|-r\right)^2=\min _{\|\mathbf{d}\|=r}\left\|\left(\mathbf{p}_1-\mathbf{p}_2\right)-\mathbf{d}\right\|^2
$$


Geometric meaning:
- $\mathbf{d}$ is a vector of length $r$.
- They "shift" $\mathbf{p}_1-\mathbf{p}_2$ by $\mathbf{d}$ to try to bring the length from $\left\|\mathbf{p}_1-\mathbf{p}_2\right\|$ closer to $r$.
- The optimal $\mathbf{d}$ is basically $\mathbf{d}=\frac{r}{\left\|\mathbf{p}_1-\mathbf{p}_2\right\|}\left(\mathbf{p}_1-\mathbf{p}_2\right)$ (the same direction, but scaled to length $r$ ).

They prove it by:
1. Showing $\left\|\mathbf{p}_1-\mathbf{p}_2\right\|-r \leq\left\|\mathbf{p}_1-\mathbf{p}_2-\mathbf{d}\right\|$ (reverse triangle inequality).
2. Substituting the specific vector $\mathbf{d}$ that achieves equality.

Hence, the spring potential can be viewed as the optimal "shift" of $\mathbf{p}_1-\mathbf{p}_2$ onto a sphere of radius $r$.

"new slide"

2. Summing Over All Springs \& Matrix Form

They then sum over all springs (indexed by $i=1, \ldots, s$ ). Let each spring connect points with indices $\left(i_1, i_2\right)$. We define:

$$
\mathbf{p}_{i_1}, \mathbf{p}_{i_2} \quad \longrightarrow \quad \text { endpoints of the } i \text {-th spring. }
$$


In vector form, collect all positions as

$$
\mathbf{x}=\left(\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_m\right) \in \mathbb{R}^{3 m}
$$


They define $\mathbf{d}_i \in \mathbb{R}^3$ for the $i$-th spring, with the constraint $\left\|\mathbf{d}_i\right\|=r_i$.
Matrix L (the Laplacian)
They rewrite

$$
\frac{1}{2} \sum_{i=1}^s k_i\left\|\mathbf{p}_{i_1}-\mathbf{p}_{i_2}-\mathbf{d}_i\right\|^2
$$

in a form

$$
\frac{1}{2} \mathbf{x}^{\top} \mathbf{L} \mathbf{x}-\mathbf{x}^{\top} \mathbf{J} \mathbf{d}
$$


The matrix $\mathbf{L}$ is (up to the factor $k_i$ ) the graph Laplacian for the mass-spring system, extended into 3D via a Kronecker product with the $3 \times 3$ identity. Concretely:
- For each spring $i$ connecting vertices (particles) $i_1$ and $i_2$, we define a vector $\mathbf{A}_i \in \mathbb{R}^m$ with entries

$$
A_{i, i_1}=+1, \quad A_{i, i_2}=-1, \quad \text { and } 0 \text { otherwise. }
$$


This "incidence vector" basically picks out the difference $\mathbf{p}_{i_1}-\mathbf{p}_{i_2}$.

Matrix $L$

$L = (\sum_{i=1}^s k_iA_i A_i^T) \otimes I_3$.

Usually, the scalar Laplacian matrix $L$ of a graph is $\sum_{i=1}^s k_i A_i A_i^{\top}$. Each edge (spring) contributes an outer product of its incidence vector, weighted by the spring stiffness $k_i$. The vector dimension is $m$ in the scalar case, but we replicate it 3 times for the 3D positions.

Matrix J
- J is also formed by combining these incidence vectors with the auxiliary "spring-indicator" vectors $\mathbf{S}_i$.
- $\mathbf{S}_i \in \mathbb{R}^s$ has entry 1 for the $i$-th spring and 0 otherwise.
- Putting it all together yields the block structure so that $\mathbf{x}^{\top} \mathbf{J d}$ collects terms of the form  $(p_{i_1}p_{i_2})\cdot \mathbf{d}_i$.

The upshot is:

$$
\sum_{i=1}^s \frac{1}{2} k_i\left\|\mathbf{p}_{i_1}-\mathbf{p}_{i_2}-\mathbf{d}_i\right\|^2=\frac{1}{2} \mathbf{x}^{\top} \mathbf{L} \mathbf{x}-\mathbf{x}^{\top} \mathbf{J} \mathbf{d}+(\text { constant in } \mathbf{x} \text { and } \mathbf{d})
$$

(The constant term would be $\frac{1}{2} \sum_i k_i\left\|\mathbf{d}_i\right\|^2$, often omitted since it doesn't affect the gradients.)
"new page":
Code helper for the matrixes:
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
- L represents the stiffness interactions between particles. Diagonal blocks represent "self-stiffness," while off-diagonal blocks represent "coupling stiffness" between connected particles.
- J represents the spring directions $\mathbf{d}_i$ to the particle displacements. Each column block ensures that the direction of a spring influences only its connected particles.

"new page"

3. Adding External Forces and Defining $E$ (x)

They now add external forces $\mathbf{f}_{\text {ext }} \in \mathbb{R}^{3 m}$. Since a force corresponds to a potential term - $\mathbf{f}_{\text {ext }} \cdot \mathbf{x}$ (or $+\mathbf{x}^{\top} \mathbf{f}_{\text {ext }}$ ), we get:

$$
E(\mathbf{x})=\min _{\mathbf{d} \in U} \frac{1}{2} \mathbf{x}^{\top} \mathbf{L} \mathbf{x}-\mathbf{x}^{\top} \mathbf{J} \mathbf{d}+\mathbf{x}^{\top} \mathbf{f}_{\text {ext }}
$$

where

$$
U=\left\{\left(\mathbf{d}_1, \ldots, \mathbf{d}_s\right) \in \mathbb{R}^{3 s}:\left\|\mathbf{d}_i\right\|=r_i \text { for each } i\right\}
$$

(The set $U$ enforces that each $\mathbf{d}_i$ has magnitude $r_i$.)
This expression says: "To evaluate the spring energy plus external-force energy at a particular $\mathbf{x}$, we pick the best $\mathbf{d} \in U$ that minimizes the sum of squared spring lengths (via the 'shift' approach)."
4. Final Objective Including Implicit Euler (Equation 14)

Recall from Section 3 that implicit Euler leads to minimizing:

$$
g(\mathbf{x})=\frac{1}{2}(\mathbf{x}-\mathbf{y})^{\top} \mathbf{M}(\mathbf{x}-\mathbf{y})+h^2 E(\mathbf{x})
$$

where $\mathbf{y}$ is a known vector collecting the old time-step data (and $\mathbf{M}$ is the diagonal mass matrix).
So if we replace $E(\mathbf{x})$ by the new expression that includes the minimum over $\mathbf{d}$:

$$
\min _{\mathbf{x}, \mathbf{d} \in U} \frac{1}{2}(\mathbf{x}-\mathbf{y})^{\top} \mathbf{M}(\mathbf{x}-\mathbf{y})+h^2\left(\frac{1}{2} \mathbf{x}^{\top} \mathbf{L} \mathbf{x}-\mathbf{x}^{\top} \mathbf{J} \mathbf{d}+\mathbf{x}^{\top} \mathbf{f}_{e x t}\right)
$$


it written in a slightly rearranged form, like

$$
\min _{\mathbf{x}, \mathrm{d} \in U} \frac{1}{2} \mathbf{x}^{\top}\left(\mathbf{M}+h^2 \mathbf{L}\right) \mathbf{x}-h^2 \mathbf{x}^{\top} \mathbf{J} \mathbf{d}+\mathbf{x}^{\top} \mathbf{b}
$$

where $\mathbf{b}$ collects $\mathrm{f}_{\text {ext }}$ and the known inertia terms.

In practice, with gravity alone the energy objective becomes a quadratic function in $\mathbf{x}$ once the spring directions $\mathbf{d}$ are fixed. Its form is

$$
\frac{1}{2} \mathbf{x}^{\top}\left(\mathbf{M}+h^2 \mathbf{L}\right) \mathbf{x}-h^2 \mathbf{x}^{\top} \mathbf{J} \mathbf{d}+\mathbf{x}^{\top} \mathbf{b}
$$

where now $\mathbf{b}$ is completely determined by the known inertia and the gravitational force. This is a standard quadratic least-squares problem (LSP) solve be solved by:

$$
\left(\mathbf{M}+h^2 \mathbf{L}\right) \mathbf{x}=h^2 \mathbf{J d}-\mathbf{b}
$$


## derivite
Denote the updation func: 
$$
A q_{n+1}=d
$$

where
- $A=M+h^2 L(k)$ (with $L$ depending on the stiffness parameters $k$ ),
- $d=h^2 J(k) \mathbf{d}_s-b$ and
- $b=-M\left(2 q_n-q_{n-1}\right)+h^2 f_{\text {grav }}$ (with
$
f_{\text {grav }}=\left(\begin{array}{c}
m_1 g \\
m_2 g \\
\vdots \\
m_m g
\end{array}\right) \text {). }
$


In this update, $q_{n+1}$ depends on $k$ in two ways:
1. Directly, through the matrices $A$ and $d$ (since $L$ and $J$ depend on $k$ ).
2. Indirectly, because the previous states $q_n$ and $q_{n-1}$ (which appear in $b$ ) also depend on $k$.


When we treat $q_n$ and $q_{n-1}$ as fixed, differentiating

$$
q_{n+1}=A^{-1}(k) d(k)
$$

with respect to a particular stiffness $k_i$ gives

$$
\frac{\partial q_{n+1}}{\partial k_i}=-A^{-1} \frac{\partial A}{\partial k_i} q_{n+1}+A^{-1} \frac{\partial d}{\partial k_i}
$$

with
- The derivative of $A$ is

$$
\frac{\partial A}{\partial k_i}=h^2\left(\left(A_i A_i^{\top}\right) \otimes I_3\right)
$$

- The derivative of $d$ is

$$
\frac{\partial d}{\partial k_i}=h^2\left(\left(A_i S_i^{\top}\right) \otimes I_3\right) \mathbf{d}_s
$$

where $A_i$ is the incidence vector for spring $i$ and $S_i$ is the corresponding indicator vector.

### Total Recursive Including $q_n$ and $q_{n-1}$

Because $q_n$ and $q_{n-1}$ also depend on $k$, putting everything together, for each stiffness parameter $k_i$, the total derivative is

$$
\frac{d q_{n+1}}{d k_i}=\underbrace{\left[-A^{-1} \frac{\partial A}{\partial k_i} q_{n+1}+A^{-1} \frac{\partial d}{\partial k_i}\right]}_{\text { direct part}}+\underbrace{2 A^{-1} M \frac{d q_n}{d k_i}}_{\text {from } q_n}-\underbrace{A^{-1} M \frac{d q_{n-1}}{d k_i}}_{\text {from } q_{n-1}} .
$$


For all parameters $k$, we can write

$$
\frac{d q_{n+1}}{d k}=-A^{-1} \frac{\partial A}{\partial k} q_{n+1}+A^{-1} \frac{\partial d}{\partial k}+2 A^{-1} M \frac{d q_n}{d k}-A^{-1} M \frac{d q_{n-1}}{d k}
$$



# remove y (the acceleration part)
# determine the boundary condition
# rewrite the mesh file