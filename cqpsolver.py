from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Problem:
    """Convex quadratic program class to store problem parameters."""

    Q: sp.csc_array  # (n,n)
    q: np.ndarray  # (n,1)
    G: sp.csc_array  # (p,n)
    h: np.ndarray  # (p,1)
    A: sp.csc_array  # (m,n)
    b: np.ndarray  # (m,1)

    @property
    def n(self) -> int:
        """Dimension of q vector."""
        return self.q.size

    @property
    def p(self) -> int:
        """Dimension of h vector."""
        return self.h.size

    @property
    def m(self) -> int:
        """Dimension of b vector."""
        return self.b.size

@dataclass(frozen=True)
class Residuals:
    """Stores the residuals and evaluates whether convergence is reached."""

    primal_ineq: float
    primal_eq: float
    stationarity: float
    duality: float

    def converged(self, tol: float) -> bool:
        """Check residuals, duality gap, and stationarity to evaluate convergence."""
        return max(self.primal_ineq, self.primal_eq, self.stationarity, self.duality) < tol

@dataclass(frozen=True)
class SolverState:
    """Stores the state of the solver and relevant data at each iteration."""

    iter: int
    obj: float
    x: np.ndarray
    s: np.ndarray
    z: np.ndarray
    y: np.ndarray
    residuals: Residuals
    step_size: float | None

@dataclass(frozen=True)
class Solver:
    """Primal-dual interior point method solver for convex QPs."""

    prob: Problem
    tol: float = 1e-8
    max_iter: int = 25

    def find_initial_state(self) -> SolverState:
        """Calculate the initial point (x0, s0, y0, z0)."""
        # Set up and solve linear system
        LHS_row1: sp.csc_array = sp.hstack([self.prob.Q, self.prob.G.T, self.prob.A.T])
        LHS_row2: sp.csc_array = sp.hstack(
            [self.prob.G, -sp.eye(self.prob.p), sp.csc_array((self.prob.p, self.prob.m))],
        )
        LHS_row3: sp.csc_array = sp.hstack(
            [
                self.prob.A,
                sp.csc_array((self.prob.m, self.prob.p)),
                sp.csc_array((self.prob.m, self.prob.m)),
            ],
        )
        LHS: sp.csc_array = sp.vstack([LHS_row1, LHS_row2, LHS_row3], format="csc")

        RHS: np.ndarray = np.vstack([-self.prob.q, self.prob.h, self.prob.b])

        sol: np.ndarray = (sp.linalg.spsolve(LHS, RHS)).reshape(-1, 1)
        x: np.ndarray = sol[: self.prob.n]
        y: np.ndarray = sol[-self.prob.m :] if self.prob.m > 0 else np.zeros((0, 1))

        # Set initial primal/dual variables
        x0: np.ndarray = x
        y0: np.ndarray = y
        z: np.ndarray = self.prob.G @ x - self.prob.h
        alpha_p: float = np.max(z)
        s0: np.ndarray = -z if alpha_p < 0 else -z + 1 + alpha_p

        alpha_d: float = -np.min(z)
        z0: np.ndarray = z if alpha_d < 0 else z + 1 + alpha_d

        initial_obj: float = (0.5 * (x0.T @ (self.prob.Q @ x0)) + self.prob.q.T @ x0).flatten()[0]

        initial_res: Residuals = self.calc_residuals(x0, s0, z0, y0)

        initial_state: SolverState = SolverState(
            iter=0, obj=initial_obj, x=x0, s=s0, z=z0, y=y0, residuals=initial_res, step_size=None,
        )

        return initial_state

    def step(self, state: SolverState) -> SolverState:
        """Compute next iteration."""
        # Setup and solve KKT system for affine scaling directions
        LHS_row1: sp.csc_array = sp.hstack(
            [
                self.prob.Q,
                sp.csc_array((self.prob.n, self.prob.p)),
                self.prob.G.T,
                self.prob.A.T,
            ],
        )
        LHS_row2: sp.csc_array = sp.hstack(
            [
                sp.csc_array((self.prob.p, self.prob.n)),
                sp.diags_array(state.z.flatten()),
                sp.diags_array(state.s.flatten()),
                sp.csc_array((self.prob.p, self.prob.m)),
            ],
        )
        LHS_row3: sp.csc_array = sp.hstack(
            [
                self.prob.G,
                sp.eye(self.prob.p),
                sp.csc_array((self.prob.p, self.prob.p)),
                sp.csc_array((self.prob.p, self.prob.m)),
            ],
        )
        LHS_row4: sp.csc_array = sp.hstack(
            [
                self.prob.A,
                sp.csc_array((self.prob.m, self.prob.p)),
                sp.csc_array((self.prob.m, self.prob.p)),
                sp.csc_array((self.prob.m, self.prob.m)),
            ],
        )
        LHS: sp.csc_array = sp.vstack([LHS_row1, LHS_row2, LHS_row3, LHS_row4], format="csc")

        RHS_aff_row1: np.ndarray = -(
            self.prob.Q @ state.x + self.prob.q + self.prob.G.T @ state.z + self.prob.A.T @ state.y
        )
        RHS_aff_row2: np.ndarray = -state.s * state.z
        RHS_aff_row3: np.ndarray = -(self.prob.G @ state.x + state.s - self.prob.h)
        RHS_aff_row4: np.ndarray = -(self.prob.A @ state.x - self.prob.b)
        RHS_aff: np.ndarray = np.vstack([RHS_aff_row1, RHS_aff_row2, RHS_aff_row3, RHS_aff_row4])

        delta_aff: np.ndarray = (sp.linalg.spsolve(LHS, RHS_aff)).reshape(-1, 1)
        delta_s_aff: np.ndarray = delta_aff[self.prob.n : self.prob.n + self.prob.p]
        delta_z_aff: np.ndarray = delta_aff[self.prob.n + self.prob.p : self.prob.n + 2 * self.prob.p]

        # Compute centering-plus-corrector directions
        mu: np.ndarray = (state.s.T @ state.z).item() / self.prob.p

        alpha: float = min(1, self.max_step(state.s, delta_s_aff), self.max_step(state.z, delta_z_aff))
        sigma: float = (
            (state.s + alpha * delta_s_aff).T @ (state.z + alpha * delta_z_aff) / (state.s.T @ state.z).item()
        ) ** 3

        RHS_cc_row1: np.ndarray = np.zeros([self.prob.n, 1])
        RHS_cc_row2: np.ndarray = sigma * mu - delta_s_aff * delta_z_aff
        RHS_cc_row3: np.ndarray = np.zeros([self.prob.p, 1])
        RHS_cc_row4: np.ndarray = np.zeros([self.prob.m, 1])
        RHS_cc: np.ndarray = np.vstack([RHS_cc_row1, RHS_cc_row2, RHS_cc_row3, RHS_cc_row4])

        delta_cc: np.ndarray = (sp.linalg.spsolve(LHS, RHS_cc)).reshape(-1, 1)

        # Combine aff and cc directions
        delta: np.ndarray = delta_aff + delta_cc
        delta_x: np.ndarray = delta[: self.prob.n]
        delta_s: np.ndarray = delta[self.prob.n : self.prob.n + self.prob.p]
        delta_z: np.ndarray = delta[self.prob.n + self.prob.p : self.prob.n + 2 * self.prob.p]
        delta_y: np.ndarray = delta[-self.prob.m :] if self.prob.m > 0 else np.zeros((0, 1))

        # Compute step size to maintain nonnegativity of s and z
        step_size: float = min(1, 0.99 * min(self.max_step(state.s, delta_s), self.max_step(state.z, delta_z)))

        # Update primal and dual variables
        x_new: np.ndarray = state.x + step_size * delta_x
        s_new: np.ndarray = state.s + step_size * delta_s
        z_new: np.ndarray = state.z + step_size * delta_z
        y_new: np.ndarray = state.y + step_size * delta_y

        # Find updated value of objective function
        obj: float = (0.5 * (x_new.T @ (self.prob.Q @ x_new)) + self.prob.q.T @ x_new).flatten()[0]

        # Find new Residuals
        new_res: Residuals = self.calc_residuals(x_new, s_new, z_new, y_new)

        # Create new SolverState
        new_state: SolverState = SolverState(
            iter=state.iter + 1, obj=obj, x=x_new, s=s_new, z=z_new, y=y_new, residuals=new_res, step_size=step_size,
        )

        return new_state

    def solve(self) -> list[SolverState]:
        """Solve problem from start to finish, returning state history."""
        state_history: list[SolverState] = [self.find_initial_state()]

        while state_history[-1].iter < self.max_iter:
            if state_history[-1].residuals.converged(self.tol):
                break

            state_history.append(self.step(state_history[-1]))

        return state_history

    def calc_residuals(self, x: np.ndarray, s: np.ndarray, z: np.ndarray, y: np.ndarray) -> Residuals:
        """Calculate residuals given state."""
        primal_ineq: float = np.linalg.norm(self.prob.G @ x + s - self.prob.h)
        primal_eq: float = np.linalg.norm(self.prob.A @ x - self.prob.b)
        stationarity: float = np.linalg.norm(self.prob.Q @ x + self.prob.q + self.prob.G.T @ z + self.prob.A.T @ y)
        duality: float = np.linalg.norm(z.T @ s)

        return Residuals(primal_ineq, primal_eq, stationarity, duality)

    def max_step(self, v, dv) -> float:
        """Calculate max step in v direction given dv."""
        ratios = -v[dv < 0] / dv[dv < 0]
        return float(ratios.min()) if len(ratios) > 0 else float("inf")
