from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Callable

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

@dataclass(frozen=True)
class Solver:
    """Primal-dual interior point method solver for convex QPs."""

    prob: Problem
    tol: float = 1e-6
    max_iter: int = 25
    quiet: bool = False

    def __post_init__(self) -> None:
        """Initialize KKT cache."""
        object.__setattr__(self, "KKT_cache", {})

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

    def build_KKT(self, state: SolverState) -> None:
        """Build KKT system from scratch and cache it, along with s and z positions."""
        # Setup and solve KKT system for affine scaling directions
        prob = self.prob
        n = prob.n
        m = prob.m
        p = prob.p
        s = state.s
        z = state.z
        LHS_row1: sp.csc_array = sp.hstack(
            [
                prob.Q,
                sp.csc_array((n, p)),
                prob.G.T,
                prob.A.T,
            ],
        )
        LHS_row2: sp.csc_array = sp.hstack(
            [
                sp.csc_array((p, n)),
                sp.diags_array(z.flatten()),
                sp.diags_array(s.flatten()),
                sp.csc_array((p, m)),
            ],
        )
        LHS_row3: sp.csc_array = sp.hstack(
            [
                prob.G,
                sp.eye(p),
                sp.csc_array((p, p)),
                sp.csc_array((p, m)),
            ],
        )
        LHS_row4: sp.csc_array = sp.hstack(
            [
                prob.A,
                sp.csc_array((m, p)),
                sp.csc_array((m, p)),
                sp.csc_array((m, m)),
            ],
        )

        LHS: sp.csc_array = sp.vstack([LHS_row1, LHS_row2, LHS_row3, LHS_row4], format="csc")

        # Finding block where diag(z) is
        col_start_z, col_end_z = n, n + p
        ptr_start_z, ptr_end_z = LHS.indptr[col_start_z], LHS.indptr[col_end_z]

        # Find non-zero elements
        counts_z = np.diff(LHS.indptr[col_start_z : col_end_z + 1])
        cols_z = np.repeat(np.arange(col_start_z, col_end_z), counts_z)

        # Find diag, i.e. row == col, mask z block appropriately
        mask_z = LHS.indices[ptr_start_z:ptr_end_z] == cols_z
        z_data_idx = np.arange(ptr_start_z, ptr_end_z)[mask_z]

        col_start_s, col_end_s = n + p, n + 2 * p
        ptr_start_s, ptr_end_s = LHS.indptr[col_start_s], LHS.indptr[col_end_s]

        counts_s: np.ndarray = np.diff(LHS.indptr[col_start_s : col_end_s + 1])
        cols_s: np.ndarray = np.repeat(np.arange(col_start_s, col_end_s), counts_s)

        mask_s: np.ndarray = LHS.indices[ptr_start_s:ptr_end_s] == cols_s - p
        s_data_idx: np.ndarray = np.arange(ptr_start_s, ptr_end_s)[mask_s]

        # Store LHS and z/s indices in Solver cache
        cache: dict = self.KKT_cache
        cache["LHS"]: sp.csc_array = LHS
        cache["z_data_idx"]: np.ndarray = z_data_idx
        cache["s_data_idx"]: np.ndarray = s_data_idx

    def step(self, state: SolverState) -> SolverState:
        """Compute next state given current state."""
        cache: dict = self.KKT_cache
        if "LHS" not in cache:
            self.build_KKT(state)
        else:
            cache["LHS"].data[cache["z_data_idx"]]: sp.csc_array = state.z.flatten()
            cache["LHS"].data[cache["s_data_idx"]]: sp.csc_array = state.s.flatten()
        solve_KKT: Callable[[np.ndarray], np.ndarray] = sp.linalg.factorized(cache["LHS"])

        RHS_aff_row1: np.ndarray = -(
            self.prob.Q @ state.x + self.prob.q + self.prob.G.T @ state.z + self.prob.A.T @ state.y
        )
        RHS_aff_row2: np.ndarray = -state.s * state.z
        RHS_aff_row3: np.ndarray = -(self.prob.G @ state.x + state.s - self.prob.h)
        RHS_aff_row4: np.ndarray = -(self.prob.A @ state.x - self.prob.b)
        RHS_aff: np.ndarray = np.vstack([RHS_aff_row1, RHS_aff_row2, RHS_aff_row3, RHS_aff_row4])

        delta_aff: np.ndarray = (solve_KKT(RHS_aff.flatten())).reshape(-1, 1)
        delta_s_aff: np.ndarray = delta_aff[self.prob.n : self.prob.n + self.prob.p]
        delta_z_aff: np.ndarray = delta_aff[self.prob.n + self.prob.p : self.prob.n + 2 * self.prob.p]

        # Compute centering-plus-corrector directions
        mu: float = (state.s.T @ state.z).item() / self.prob.p

        alpha: float = min(1, self.max_step(state.s, delta_s_aff), self.max_step(state.z, delta_z_aff))
        sigma: float = (
            ((state.s + alpha * delta_s_aff).T @ (state.z + alpha * delta_z_aff)).item() / (state.s.T @ state.z).item()
        ) ** 3

        RHS_cc_row1: np.ndarray = np.zeros([self.prob.n, 1])
        RHS_cc_row2: np.ndarray = sigma * mu - delta_s_aff * delta_z_aff
        RHS_cc_row3: np.ndarray = np.zeros([self.prob.p, 1])
        RHS_cc_row4: np.ndarray = np.zeros([self.prob.m, 1])
        RHS_cc: np.ndarray = np.vstack([RHS_cc_row1, RHS_cc_row2, RHS_cc_row3, RHS_cc_row4])

        delta_cc: np.ndarray = (solve_KKT(RHS_cc.flatten())).reshape(-1, 1)

        # Combine aff and cc directions
        delta: np.ndarray = delta_aff + delta_cc
        delta_x: np.ndarray = delta[: self.prob.n]
        delta_s: np.ndarray = delta[self.prob.n : self.prob.n + self.prob.p]
        delta_z: np.ndarray = delta[self.prob.n + self.prob.p : self.prob.n + 2 * self.prob.p]
        delta_y: np.ndarray = delta[-self.prob.m :] if self.prob.m > 0 else np.zeros((0, 1))

        # Compute step size to maintain nonnegativity of s and z
        step_size: float = min(1, 0.9999 * min(self.max_step(state.s, delta_s), self.max_step(state.z, delta_z)))

        # Update primal and dual variables
        x_new: np.ndarray = state.x + step_size * delta_x
        s_new: np.ndarray = state.s + step_size * delta_s
        z_new: np.ndarray = state.z + step_size * delta_z
        y_new: np.ndarray = state.y + step_size * delta_y

        # Find updated value of objective function
        obj: float = (0.5 * (x_new.T @ (self.prob.Q @ x_new)) + self.prob.q.T @ x_new).item()

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
