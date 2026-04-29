# /// script
# requires-python = ">=3.13"
# dependencies = ["pandas", "matplotlib", "numpy"]
# ///
"""
Dolan-Moré performance profiles for CQPSolver.

  Relative profile: fraction of problems where t_s / t_best <= tau
  Absolute profile: fraction of problems solved within t seconds (empirical CDF)

With a single solver the relative profile trivially has tau=1 for every solved
problem, so the curve is a step at tau=1 reaching the solve rate.  The absolute
profile is the more informative plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern Roman", "font.size": 12, "figure.dpi": 200})

CSV = "mm_benchmark_results.csv"
SOLVER_NAME = "CQPSolver"
# Problems that did not converge are treated as t = inf
INF = float("inf")

# ── load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df["converged"] = (
    df["converged"].astype(str).str.strip().str.lower()
    .map({"true": True, "false": False, "1": True, "0": False})
)

n_total = len(df)

# Effective solve time: actual time if converged, inf otherwise
df["t_eff"] = df.apply(
    lambda r: r["solve_time_s"] if r["converged"] else INF, axis=1
)

# ── performance profile helpers ────────────────────────────────────────────────

def relative_profile(times: np.ndarray, n_total: int, tau_grid: np.ndarray) -> np.ndarray:
    """
    Dolan-Moré relative profile for a single solver.
    With one solver, t_best = t_s for every problem, so tau_ratio = 1 for
    all solved problems and inf for failures.
    rho(tau) = #{p : ratio_p <= tau} / n_total
    """
    # ratio is 1 for solved, inf for failed
    ratios = np.where(np.isfinite(times), 1.0, INF)
    return np.array([(ratios <= tau).mean() for tau in tau_grid])


def absolute_profile(times: np.ndarray, n_total: int, t_grid: np.ndarray) -> np.ndarray:
    """
    Fraction of ALL problems solved within wall-clock time t.
    """
    return np.array([(times <= t).sum() / n_total for t in t_grid])


times = df["t_eff"].values

# ── grids ──────────────────────────────────────────────────────────────────────
tau_grid = np.logspace(0, 3, 2000)          # 1 … 1000
t_min = df.loc[df["converged"], "solve_time_s"].min()
t_max = df.loc[df["converged"], "solve_time_s"].max()
t_grid = np.logspace(np.log10(t_min) - 0.5, 3, 4000)

rho_rel = relative_profile(times, n_total, tau_grid)
rho_abs = absolute_profile(times, n_total, t_grid)

# ── color / style ──────────────────────────────────────────────────────────────
COLOR = "#5B4FCF"   # purple, similar to Clarabel in the reference
LW = 2.0

fig, ax = plt.subplots(figsize=(7, 5))

solve_rate = df["converged"].mean()

# ── absolute performance profile ───────────────────────────────────────────────
ax.semilogx(t_grid, rho_abs, color=COLOR, lw=LW, label=SOLVER_NAME)

ax.axhline(solve_rate, color=COLOR, lw=0.8, ls="--", alpha=0.5,
           label=f"Solve rate {solve_rate:.1%}")

# mark the median solve time of converged problems
t_med = df.loc[df["converged"], "solve_time_s"].median()
frac_at_med = (times <= t_med).sum() / n_total
ax.axvline(t_med, color="gray", lw=0.9, ls=":", alpha=0.7)
ax.annotate(
    f"Median\n{t_med:.3f} s",
    xy=(t_med, frac_at_med),
    xytext=(t_med * 6, frac_at_med - 0.18),
    fontsize=8,
    color="gray",
    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
)

ax.set_xlim(t_grid[0], 1e3)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Solvetime $t$ [seconds]", fontsize=11)
ax.set_ylabel("Fraction of problems solved within $t$", fontsize=11)
ax.set_title("Solution Time Profile")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig("performance_profiles.png", dpi=150, bbox_inches="tight")
print("Saved performance_profiles.png")

# ── summary stats ──────────────────────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"  Total problems        : {n_total}")
print(f"  Converged             : {df['converged'].sum()}  ({solve_rate:.1%})")
print(f"  Failed / no-converge  : {(~df['converged']).sum()}")
print(f"  Median solve time     : {t_med:.4f} s  (converged only)")
print(f"  Mean solve time       : {df.loc[df['converged'], 'solve_time_s'].mean():.4f} s")
print(f"  Max solve time        : {df.loc[df['converged'], 'solve_time_s'].max():.4f} s")
print(f"{'='*45}")
