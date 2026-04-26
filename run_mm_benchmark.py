from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

from cqpsolver import Problem, Solver

MAT_DIR = Path("../QP-Test-Problems/MAT_Files")
OUTPUT_CSV = Path("mm_benchmark_results.csv")
SKIP: frozenset[str] = frozenset({"BOYD2"})

CSV_FIELDNAMES: list[str] = [
    "name",
    "n",
    "m",
    "p",
    "converged",
    "iters",
    "obj",
    "solve_time_s",
    "primal_ineq",
    "primal_eq",
    "stationarity",
    "duality",
    "step_size",
    "msg",
    "error",
]


@contextlib.contextmanager
def suppress_c_output():
    """Redirect stdout/stderr at the fd level to silence C-library messages (e.g. UMFPACK)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


def parse_mat(filepath: Path) -> Problem:
    mat_dict = loadmat(filepath)

    Q: sp.csc_array = sp.csc_array(mat_dict["Q"].astype(float))
    q: np.ndarray = mat_dict["c"].astype(float).reshape(-1, 1)
    A: sp.csc_array = sp.csc_array(mat_dict["A"].astype(float))
    rl: np.ndarray = mat_dict["rl"].astype(float).flatten()
    ru: np.ndarray = mat_dict["ru"].astype(float).flatten()
    lb: np.ndarray = mat_dict["lb"].astype(float).flatten().reshape(-1, 1)
    ub: np.ndarray = mat_dict["ub"].astype(float).flatten().reshape(-1, 1)

    eq_mask: np.ndarray = rl == ru
    A_eq: sp.csc_array = sp.csc_array(A[eq_mask])
    b_eq: np.ndarray = ru[eq_mask].reshape(-1, 1)
    A_eq = A_eq if A_eq.size > 0 else sp.csc_array((0, A.shape[1]))
    b_eq = b_eq if b_eq.size > 0 else np.zeros((0, 1))

    ineq_mask: np.ndarray = np.invert(eq_mask)
    G_ineq: sp.csc_array = sp.vstack([A[ineq_mask], -A[ineq_mask]], format="csc")
    h_ineq: np.ndarray = np.concatenate([ru[ineq_mask], -rl[ineq_mask]]).reshape(-1, 1)

    n: int = Q.shape[0]
    G_full: sp.csc_array = sp.vstack([G_ineq, sp.eye(n), -sp.eye(n)], format="csc")
    h_full: np.ndarray = np.vstack([h_ineq, ub, -lb])

    finite_mask: np.ndarray = np.isfinite(h_full).flatten()
    G: sp.csc_array = sp.csc_array(G_full[finite_mask])
    h: np.ndarray = h_full[finite_mask].reshape(-1, 1)

    return Problem(Q=Q, q=q, G=G, h=h, A=A_eq, b=b_eq)


def solve_problem(name: str, filepath: Path) -> dict[str, object]:
    nan = float("nan")
    row: dict[str, object] = {
        "name": name,
        "n": None,
        "m": None,
        "p": None,
        "converged": False,
        "iters": None,
        "obj": nan,
        "solve_time_s": nan,
        "primal_ineq": nan,
        "primal_eq": nan,
        "stationarity": nan,
        "duality": nan,
        "step_size": "",
        "msg": "",
        "error": "",
    }
    try:
        prob = parse_mat(filepath)
        row["n"] = prob.n
        row["m"] = prob.m
        row["p"] = prob.p

        solver = Solver(prob, max_iter=100, quiet=True, n_refine=2)
        t0 = time.perf_counter()
        with suppress_c_output():
            result, _ = solver.solve()
        row["solve_time_s"] = time.perf_counter() - t0

        fs = result.final_state
        row["converged"] = result.convergence
        row["iters"] = fs.iter
        row["obj"] = fs.obj
        row["primal_ineq"] = fs.residuals.primal_ineq
        row["primal_eq"] = fs.residuals.primal_eq
        row["stationarity"] = fs.residuals.stationarity
        row["duality"] = fs.residuals.duality
        row["step_size"] = "" if fs.step_size is None else fs.step_size
        row["msg"] = result.msg
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["msg"] = "ERROR"

    return row


def print_live_row(idx: int, total: int, row: dict[str, object]) -> None:
    prefix = f"{idx:>4}/{total}"
    name = str(row["name"])

    if row["error"]:
        print(f"{prefix}  {name:<14}  ERR   {row['error']}")
        sys.stdout.flush()
        return

    status = "OK  " if row["converged"] else "FAIL"
    n = row["n"]
    m = row["m"]
    p = row["p"]
    iters = row["iters"]
    t = row["solve_time_s"]
    obj = row["obj"]

    print(
        f"{prefix}  {name:<14}  n={n:>5}  m={m:>5}  p={p:>6}  {status}  {iters:>4} iters  {t:>8.3f}s  obj= {obj:>14.6e}",
    )
    sys.stdout.flush()


def run_benchmark(
    mat_dir: Path,
    output_csv: Path,
    skip: frozenset[str],
    problems: list[str] | None,
) -> None:
    if problems is not None:
        mat_files = sorted(
            [mat_dir / f"{name}.mat" for name in problems],
            key=lambda p: p.stem,
        )
        missing = [str(f) for f in mat_files if not f.exists()]
        if missing:
            print(f"Warning: files not found: {', '.join(missing)}", file=sys.stderr)
        mat_files = [f for f in mat_files if f.exists()]
    else:
        mat_files = sorted(mat_dir.glob("*.mat"), key=lambda p: p.stem)
        mat_files = [f for f in mat_files if f.stem not in skip]

    total = len(mat_files)
    print(f"Running {total} problems  →  {output_csv}")
    if skip and problems is None:
        print(f"Skipping: {', '.join(sorted(skip))}")
    print()

    solved = 0
    failed = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        csvfile.flush()

        try:
            for i, filepath in enumerate(mat_files, start=1):
                row = solve_problem(filepath.stem, filepath)
                print_live_row(i, total, row)
                writer.writerow(row)
                csvfile.flush()

                if row["error"]:
                    failed += 1
                elif row["converged"]:
                    solved += 1
                else:
                    failed += 1

        except KeyboardInterrupt:
            completed = i - 1 if "i" in dir() else 0
            print(f"\nInterrupted after {completed}/{total} problems.")
            print(f"Partial results saved to {output_csv}")
            raise

    print(f"\nSolved {solved}/{total}  •  Failed/ERR {failed}/{total}  •  Results in {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CQPSolver on MM QP problems")
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Comma-separated list of problem names to run (e.g. EXDATA,HS21). Omit to run all problems.",
    )
    args = parser.parse_args()

    problem_list: list[str] | None = [p.strip() for p in args.problems.split(",")] if args.problems else None

    run_benchmark(MAT_DIR, OUTPUT_CSV, SKIP, problem_list)


if __name__ == "__main__":
    main()
