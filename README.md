This project implements a primal-dual interior point method solver for convex quadratic programs in Python.

**Standard Inequality QP:**

$\underset{x}{\min} \quad \frac{1}{2} x^\top Q x + q^\top x$ 

$\text{s.t.} \quad Gx \leq h, \qquad Ax = b$

$ Q \in \mathbb{S}_+^n, \quad q \in \mathbb{R}^n$

$ G \in \mathbb{R}^{p \times n}, \quad h \in \mathbb{R}^p $

$ A \in \mathbb{R}^{m \times n}, \quad b \in \mathbb{R}^m $

**Solve by introducing a slack variable:**

$ \underset{x}{\min} \quad \frac{1}{2} x^\top Q x + q^\top x $ 

$ \text{s.t.} \quad Gx + s = h, \qquad Ax = b, \qquad s \geq 0 $

$ s \in \mathbb{R}^p $

**Dual variables:**

Dual equality variables: $ y \in \mathbb{R}^m $

Dual inequality variables: $ z \in \mathbb{R}^p $

**KKT Conditions:**

$ Gx + s = h, \qquad Ax = b, \qquad s \geq 0 $

$ z \geq 0 $

$ Qx + q + G^\top z + A^\top y = 0 $

$ z_i s_i = 0, \quad i = 1, \ldots, p $

**Main KKT system to solve:**

$$
\underbrace{\begin{bmatrix}
Q & 0 & G^\top & A^T \\
0 & Z & S & 0 \\
G & I & 0 & 0 \\
A & 0 & 0 & 0
\end{bmatrix}}_{(n + 2p + m)\times(n + 2p + m)}
\underbrace{\begin{bmatrix}
\Delta x^{\text{aff}} \\
\Delta s^{\text{aff}} \\
\Delta z^{\text{aff}} \\
\Delta y^{\text{aff}}
\end{bmatrix}}_{(n + 2p + m)\times1}
=
\underbrace{\begin{bmatrix}
-(A^\top y + G^\top z + Q x + q) \\
-S z \\
-(G x + s - h) \\
-(A x - b)
\end{bmatrix}}_{(n + 2p + m)\times1}
$$