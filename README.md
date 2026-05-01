This project implements a primal-dual interior point method solver for convex quadratic programs in Python.

## Problem Formulation

The solver handles convex quadratic programs of the form:

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & \frac{1}{2} x^\top Q x + q^\top x \\
\text{subject to} \quad & Gx \leq h \\
& Ax = b
\end{aligned}
$$

where:

| Symbol | Shape | Description |
|--------|-------|-------------|
| $Q \in \mathbb{S}^n_+$ | $n \times n$ | Symmetric positive semidefinite cost matrix |
| $q$ | $n \times 1$ | Linear cost vector |
| $G$ | $p \times n$ | Inequality constraint matrix |
| $h$ | $p \times 1$ | Inequality constraint right-hand side |
| $A$ | $m \times n$ | Equality constraint matrix |
| $b$ | $m \times 1$ | Equality constraint right-hand side |
