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

plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern Roman", "font.size": 12, "figure.dpi": 200})

CSV_PY  = "mm_benchmark_results.csv"
CSV_CPP = "mm_benchmark_results_cpp.csv"
INF = float("inf")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["converged"] = (
        df["converged"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    df["t_eff"] = df.apply(lambda r: r["solve_time_s"] if r["converged"] else INF, axis=1)
    return df


def absolute_profile(times: np.ndarray, n_total: int, t_grid: np.ndarray) -> np.ndarray:
    return np.array([(times <= t).sum() / n_total for t in t_grid])


df_py  = load(CSV_PY)
df_cpp = load(CSV_CPP)

n_total = len(df_py)

solvers = [
    ("Python",  df_py,  "#5B4FCF"),
    ("C++",     df_cpp, "#D95F02"),
]

# ── shared time grid spanning both solvers' converged times ───────────────────
t_min = min(
    df.loc[df["converged"], "solve_time_s"].min()
    for _, df, _ in solvers
)
t_max = max(
    df.loc[df["converged"], "solve_time_s"].max()
    for _, df, _ in solvers
)
t_grid = np.logspace(np.log10(t_min) - 0.5, np.log10(t_max) + 0.5, 4000)

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for name, df, color in solvers:
    times = df["t_eff"].values
    rho   = absolute_profile(times, n_total, t_grid)
    solve_rate = df["converged"].mean()

    pct = f"{solve_rate * 100:.1f}\\%"
    ax.semilogx(t_grid, rho, color=color, lw=2.0, label=f"{name} ({pct} solved)")
    ax.axhline(solve_rate, color=color, lw=0.8, ls="--", alpha=0.4)



ax.set_xlim(t_grid[0], t_grid[-1])
ax.set_ylim(0, 1.05)
ax.set_xlabel("Solvetime $t$ [seconds]", fontsize=11)
ax.set_ylabel("Fraction of problems solved within $t$", fontsize=11)
ax.set_title(r"Absolute Performance Profile: Python vs.\ C++")
leg = ax.legend(fontsize=9, loc="lower right", borderaxespad=1.0)
leg.set_clip_on(False)
ax.grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig("performance_profiles.png", dpi=150, bbox_inches="tight")
print("Saved performance_profiles.png")

# ── summary stats ──────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"{'':20s} {'Python':>15s} {'C++':>15s}")
print(f"{'='*55}")
for label, attr, fmt in [
    ("Total problems",   None,             "d"),
    ("Converged",        "converged",      "d"),
    ("Solve rate",       "converged_rate", ".1%"),
    ("Median time (s)",  "t_med",          ".4f"),
    ("Mean time (s)",    "t_mean",         ".4f"),
    ("Max time (s)",     "t_max",          ".4f"),
]:
    vals = []
    for _, df, _ in solvers:
        conv = df["converged"]
        t    = df.loc[conv, "solve_time_s"]
        if attr is None:
            vals.append(f"{len(df):d}")
        elif attr == "converged":
            vals.append(f"{conv.sum():d}")
        elif attr == "converged_rate":
            vals.append(f"{conv.mean():.1%}")
        elif attr == "t_med":
            vals.append(f"{t.median():.4f}")
        elif attr == "t_mean":
            vals.append(f"{t.mean():.4f}")
        elif attr == "t_max":
            vals.append(f"{t.max():.4f}")
    print(f"  {label:<20s} {vals[0]:>15s} {vals[1]:>15s}")
print(f"{'='*55}")
