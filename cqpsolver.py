from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sksparse.umfpack import UMFFactor, umf_factor

header: str = (
    f"{'Iter.':^5} │ {'Objective':^14} │ {'Primal Inequality':^17} │ "
    f"{'Primal Equality':^15} │ {'Stationarity':^12} │ {'Duality':^11} │ {'Step Size':^6}"
)

divider: str = "─" * len(header)

@dataclass(frozen=True)
class Problem:
    """Class to store convex QP problem parameters."""

    Q: sp.csc_array  # (n,n)
    q: np.ndarray  # (n,1)
    G: sp.csc_array  # (p,n)
    h: np.ndarray  # (p,1)
    A: sp.csc_array  # (m,n)
    b: np.ndarray  # (m,1)

    @property
    def n(self) -> int:
        """Size of q vector."""
        return self.q.size

    @property
    def p(self) -> int:
        """Size of h vector."""
        return self.h.size

    @property
    def m(self) -> int:
        """Size of b vector."""
        return self.b.size

@dataclass(frozen=True)
class Residuals:
    """Stores the residuals for a state."""

    primal_ineq: float
    primal_eq: float
    stationarity: float
    duality: float

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
class Result:
    """Stores final result of solver."""

    convergence: bool
    msg: str
    final_state: SolverState

@dataclass
class Solver:
    """Primal-dual interior point method solver for convex QPs."""

    prob: Problem
    tol: float = 1e-8
    max_iter: int = 25
    quiet: bool = False
    _LHS_solver: UMFFactor | None = field(default=None, init=False, repr=False)

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

        initial_obj: float = (0.5 * (x0.T @ (self.prob.Q @ x0)) + self.prob.q.T @ x0).item()

        initial_res: Residuals = self.calc_residuals(x0, s0, z0, y0)

        initial_state: SolverState = SolverState(
            iter=0, obj=initial_obj, x=x0, s=s0, z=z0, y=y0, residuals=initial_res, step_size=None,
        )

        if not self.quiet:
            self._print_header()
            self._print_row(initial_state)

        return initial_state

    def step(self, state: SolverState) -> SolverState:
        """Compute next state given current state."""
        prob: Problem = self.prob
        n, m, p = prob.n, prob.m, prob.p
        s, z = state.s.flatten(), state.z.flatten()
        zs: np.ndarray = z / s

        # Static regularization value
        delta = 1e-8

        # Build reduced (n+m) x (n+m) system once per iteration
        GTzsG: sp.csc_array = prob.G.T @ sp.diags_array(zs) @ prob.G
        LHS_11: sp.csc_array = prob.Q + GTzsG

        # Unregularzed LHS for iterative refinement
        LHS_orig: sp.csc_array = sp.block_array(
            [[LHS_11, prob.A.T], [prob.A, sp.csc_array((m, m))]], format="csc",
        ) if m > 0 else LHS_11

        # Regularized LHS
        LHS: sp.csc_array = sp.block_array(
            [[LHS_11 + delta * sp.eye(n), prob.A.T], [prob.A, sp.csc_array((m, m))]], format="csc",
        ) if m > 0 else LHS_11 + delta * sp.eye(n)

        if self._LHS_solver is None:
            self._LHS_solver = umf_factor(LHS)
        else:
            self._LHS_solver.factorize(LHS)

        def solve_reduced(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, r4: np.ndarray) -> tuple[np.ndarray]:
            """Solve reduced KKT system with iterative refinement and return solution to original system."""
            rhs_x: np.ndarray = r1 - prob.G.T @ ((r2 - z.reshape(-1, 1) * r3) / s.reshape(-1, 1))
            rhs: np.ndarray = np.vstack([rhs_x, r4]) if prob.m > 0 else rhs_x

            sol: np.ndarray = self._LHS_solver.solve(rhs.flatten()).reshape(-1, 1)

            # Iterative refinement of regularized solution
            for _ in range(2):
                residual: np.ndarray = rhs - LHS_orig @ sol
                correction: np.ndarray = self._LHS_solver.solve(residual.flatten()).reshape(-1, 1)
                sol: np.ndarray = sol + correction

            dx: np.ndarray = sol[:n]
            dy: np.ndarray = sol[n:] if prob.m > 0 else np.zeros((0, 1))
            ds: np.ndarray = r3 - prob.G @ dx
            dz: np.ndarray = (r2 - z.reshape(-1, 1) * ds) / s.reshape(-1, 1)

            return dx, ds, dz, dy

        # Affine scaling RHS
        r1_aff: np.ndarray = -(prob.Q @ state.x + prob.q + prob.G.T @ state.z + prob.A.T @ state.y)
        r2_aff: np.ndarray = -state.s * state.z
        r3_aff: np.ndarray = -(prob.G @ state.x + state.s - prob.h)
        r4_aff: np.ndarray = -(prob.A @ state.x - prob.b)

        dx_aff, ds_aff, dz_aff, dy_aff = solve_reduced(r1_aff, r2_aff, r3_aff, r4_aff)

        # Centering-corrector RHS
        mu: float = (state.s.T @ state.z).item() / p
        alpha: float = min(1, self.max_step(state.s, ds_aff), self.max_step(state.z, dz_aff))
        sigma: float = (((state.s + alpha * ds_aff).T @ (state.z + alpha * dz_aff)) / (state.s.T @ state.z)).item() ** 3

        r2_cc: np.ndarray = sigma * mu - ds_aff * dz_aff

        dx_cc, ds_cc, dz_cc, dy_cc = solve_reduced(np.zeros((n, 1)), r2_cc, np.zeros((p, 1)), np.zeros((m, 1)))

        # Combine aff and cc directions
        dx: np.ndarray = dx_aff + dx_cc
        ds: np.ndarray = ds_aff + ds_cc
        dz: np.ndarray = dz_aff + dz_cc
        dy: np.ndarray = dy_aff + dy_cc

        # Compute step size to maintain nonnegativity of s and z
        step_size: float = min(1, 0.99 * min(self.max_step(state.s, ds), self.max_step(state.z, dz)))

        # Update primal and dual variables
        x_new: np.ndarray = state.x + step_size * dx
        s_new: np.ndarray = state.s + step_size * ds
        z_new: np.ndarray = state.z + step_size * dz
        y_new: np.ndarray = state.y + step_size * dy

        # Find updated value of objective function
        obj: float = (0.5 * (x_new.T @ (prob.Q @ x_new)) + prob.q.T @ x_new).item()

        # Find new Residuals
        new_res: Residuals = self.calc_residuals(x_new, s_new, z_new, y_new)

        # Create new SolverState
        new_state: SolverState = SolverState(state.iter + 1, obj, x_new, s_new, z_new, y_new, new_res, step_size)

        if not self.quiet:
            self._print_row(new_state)

        return new_state

    def solve(self) -> tuple[Result, list[SolverState]]:
        """Solve problem from start to finish, returning state history."""
        try:
            state_history: list[SolverState] = [self.find_initial_state()]
        except Exception as e:
            convergence: bool = False
            msg: str = "Failed to find initial state, error: " + str(e)
            final_state: SolverState = SolverState(0, 0.0, 0, 0, 0, 0, Residuals(0.0, 0.0, 0.0, 0.0), 0.0)

            if not self.quiet:
                print(msg)

            return (Result(convergence, msg, final_state), [])

        try:
            while state_history[-1].iter < self.max_iter:
                if self.converged(state_history[-1]):
                    break

                state_history.append(self.step(state_history[-1]))
        except Exception as e:
            convergence: bool = False
            msg: str = "Failed while solving, error: " + str(e)
            final_state: SolverState = state_history[-1]

            if not self.quiet:
                print(divider)
                print(msg)

            return (Result(convergence, msg, final_state), state_history)

        final_state: SolverState = state_history[-1]
        convergence: bool = self.converged(final_state)
        if convergence:
            msg = f"Solved in {final_state.iter} iterations, objective value = {final_state.obj:.8g}."
        else:
            msg = "Failed to converge before iteration limit."

        result: Result = Result(convergence, msg, final_state)

        if not self.quiet:
            print(divider)
            print(msg)

        return (result, state_history)

    def calc_residuals(self, x: np.ndarray, s: np.ndarray, z: np.ndarray, y: np.ndarray) -> Residuals:
        """Calculate residuals given state."""
        prob: Problem = self.prob
        primal_ineq: float = np.linalg.norm(prob.G @ x + s - prob.h, np.inf)
        primal_eq: float = np.linalg.norm(prob.A @ x - prob.b, np.inf)
        stationarity: float = np.linalg.norm(prob.Q @ x + prob.q + prob.G.T @ z + prob.A.T @ y, np.inf)
        duality: float = abs(z.T @ s).item()

        return Residuals(primal_ineq, primal_eq, stationarity, duality)

    def max_step(self, v: np.ndarray, dv: np.ndarray) -> float:
        """Calculate max step in v direction given dv."""
        ratios: np.ndarray = -v[dv < 0] / dv[dv < 0]
        return float(ratios.min()) if len(ratios) > 0 else float("inf")

    def converged(self, state: SolverState) -> bool:
        """Check residuals, duality gap, and stationarity to evaluate convergence."""
        prob: Problem = self.prob
        res: Residuals = state.residuals

        primal_feas_check: bool = max(res.primal_ineq, res.primal_eq) < self.tol * (
            1 + max(
                np.linalg.norm(prob.A @ state.x, np.inf),
                np.linalg.norm(prob.b, np.inf),
                np.linalg.norm(prob.G @ state.x, np.inf),
                np.linalg.norm(prob.h, np.inf),
                np.linalg.norm(state.s, np.inf),
            )
        )

        stationarity_check: bool = res.stationarity < self.tol * (
            1 + max(
                np.linalg.norm(prob.Q @ state.x, np.inf),
                np.linalg.norm(prob.A.T @ state.y, np.inf),
                np.linalg.norm(prob.G.T @ state.z, np.inf),
                np.linalg.norm(prob.q, np.inf),
            )
        )

        duality_check: bool = res.duality < self.tol * (
            1 + max(
                1,
                abs((0.5 * state.x.T @ (prob.Q @ state.x) + prob.q.T @ state.x).item()),
                abs((-0.5 * state.x.T @ (prob.Q @ state.x) - prob.b.T @ state.y - prob.h.T @ state.z).item()),
            )
        )

        return all([primal_feas_check, stationarity_check, duality_check])

    def _print_header(self) -> None:
        print(divider)
        print(header)
        print(divider)

    def _print_row(self, state: SolverState) -> None:
        res = state.residuals
        step = f"{state.step_size:^9.4f}" if state.step_size is not None else f"{'—':^9}"
        print(
            f"{state.iter:^5} │ {state.obj:^14.8g} │ {res.primal_ineq:^17.4e} │ "
            f"{res.primal_eq:^15.4e} │ {res.stationarity:^12.4e} │ {res.duality:^11.4e} │ {step}",
        )
