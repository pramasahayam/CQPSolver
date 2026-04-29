# /// script
# requires-python = ">=3.13"
# dependencies = ["pandas", "matplotlib", "scipy", "numpy"]
# ///
"""
Visualize CQPSolver performance as a function of problem size (n, m, p).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

CSV = "mm_benchmark_results.csv"

df = pd.read_csv(CSV)
df["converged"] = df["converged"].astype(str).str.strip().str.lower().map(
    {"true": True, "false": False, "1": True, "0": False}
)
df["total_size"] = df["n"] + df["m"] + df["p"]
df["log_n"] = np.log10(df["n"].clip(lower=1))
df["log_m"] = np.log10(df["m"].clip(lower=1))
df["log_p"] = np.log10(df["p"].clip(lower=1))
df["log_size"] = np.log10(df["total_size"].clip(lower=1))
df["log_t"] = np.log10(df["solve_time_s"].clip(lower=1e-6))

ok = df[df["converged"] == True].copy()
fail = df[df["converged"] == False].copy()

# ── color helpers ─────────────────────────────────────────────────────────────
C_OK   = "#2c7bb6"
C_FAIL = "#d7191c"
ALPHA  = 0.75
S      = 40

fig = plt.figure(figsize=(16, 12))
fig.suptitle("CQPSolver performance vs. problem size", fontsize=15, fontweight="bold", y=0.98)

# ── 1. log-log: time vs n ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(ok["n"], ok["solve_time_s"], s=S, color=C_OK, alpha=ALPHA, label="Converged", zorder=3)
ax1.scatter(fail["n"], fail["solve_time_s"], s=S, marker="x", color=C_FAIL, alpha=ALPHA, label="Failed", zorder=3)
# power-law fit on converged
slope1, intercept1, r1, *_ = stats.linregress(ok["log_n"], ok["log_t"])
xfit = np.logspace(ok["log_n"].min(), ok["log_n"].max(), 200)
ax1.plot(xfit, 10**intercept1 * xfit**slope1, "k--", lw=1.2, label=f"fit: $t \\propto n^{{{slope1:.2f}}}$  ($R^2$={r1**2:.2f})")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("n  (variables)"); ax1.set_ylabel("Solve time (s)")
ax1.set_title("Time vs n")
ax1.legend(fontsize=8); ax1.grid(True, which="both", ls=":", alpha=0.5)

# ── 2. log-log: time vs m ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
m_ok = ok[ok["m"] > 0]; m_fail = fail[fail["m"] > 0]
ax2.scatter(m_ok["m"], m_ok["solve_time_s"], s=S, color=C_OK, alpha=ALPHA, label="Converged", zorder=3)
ax2.scatter(m_fail["m"], m_fail["solve_time_s"], s=S, marker="x", color=C_FAIL, alpha=ALPHA, label="Failed", zorder=3)
if len(m_ok) > 2:
    slope2, intercept2, r2, *_ = stats.linregress(m_ok["log_m"], m_ok["log_t"])
    xfit2 = np.logspace(m_ok["log_m"].min(), m_ok["log_m"].max(), 200)
    ax2.plot(xfit2, 10**intercept2 * xfit2**slope2, "k--", lw=1.2, label=f"fit: $t \\propto m^{{{slope2:.2f}}}$  ($R^2$={r2**2:.2f})")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("m  (equality constraints)"); ax2.set_ylabel("Solve time (s)")
ax2.set_title("Time vs m  (m > 0 only)")
ax2.legend(fontsize=8); ax2.grid(True, which="both", ls=":", alpha=0.5)

# ── 3. log-log: time vs p ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
p_ok = ok[ok["p"] > 0]; p_fail = fail[fail["p"] > 0]
ax3.scatter(p_ok["p"], p_ok["solve_time_s"], s=S, color=C_OK, alpha=ALPHA, label="Converged", zorder=3)
ax3.scatter(p_fail["p"], p_fail["solve_time_s"], s=S, marker="x", color=C_FAIL, alpha=ALPHA, label="Failed", zorder=3)
if len(p_ok) > 2:
    slope3, intercept3, r3, *_ = stats.linregress(p_ok["log_p"], p_ok["log_t"])
    xfit3 = np.logspace(p_ok["log_p"].min(), p_ok["log_p"].max(), 200)
    ax3.plot(xfit3, 10**intercept3 * xfit3**slope3, "k--", lw=1.2, label=f"fit: $t \\propto p^{{{slope3:.2f}}}$  ($R^2$={r3**2:.2f})")
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.set_xlabel("p  (inequality constraints)"); ax3.set_ylabel("Solve time (s)")
ax3.set_title("Time vs p  (p > 0 only)")
ax3.legend(fontsize=8); ax3.grid(True, which="both", ls=":", alpha=0.5)

# ── 4. log-log: time vs total size n+m+p, colored by n ───────────────────────
ax4 = fig.add_subplot(2, 3, 4)
sc = ax4.scatter(ok["total_size"], ok["solve_time_s"], s=S, c=np.log10(ok["n"].clip(1)),
                 cmap="viridis", alpha=ALPHA, zorder=3, label="Converged")
ax4.scatter(fail["total_size"], fail["solve_time_s"], s=S+10, marker="x", color=C_FAIL,
            alpha=ALPHA, zorder=4, label="Failed")
slope4, intercept4, r4, *_ = stats.linregress(ok["log_size"], ok["log_t"])
xfit4 = np.logspace(ok["log_size"].min(), ok["log_size"].max(), 200)
ax4.plot(xfit4, 10**intercept4 * xfit4**slope4, "k--", lw=1.2,
         label=f"fit: $t \\propto N^{{{slope4:.2f}}}$  ($R^2$={r4**2:.2f})")
plt.colorbar(sc, ax=ax4, label="log₁₀(n)")
ax4.set_xscale("log"); ax4.set_yscale("log")
ax4.set_xlabel("N = n + m + p  (total size)"); ax4.set_ylabel("Solve time (s)")
ax4.set_title("Time vs total size  (color = log n)")
ax4.legend(fontsize=8); ax4.grid(True, which="both", ls=":", alpha=0.5)

# ── 5. Iters distribution (converged only) ────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
bins = range(0, int(ok["iters"].max()) + 2)
ax5.hist(ok["iters"], bins=bins, color=C_OK, edgecolor="white", linewidth=0.5, alpha=0.85)
ax5.set_xlabel("Iterations to convergence"); ax5.set_ylabel("Count")
ax5.set_title("Iteration count (converged problems)")
ax5.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax5.grid(True, axis="y", ls=":", alpha=0.5)

# ── 6. Time / iter vs total size — isolates per-iteration cost ────────────────
ax6 = fig.add_subplot(2, 3, 6)
ok2 = ok[ok["iters"] > 0].copy()
ok2["time_per_iter"] = ok2["solve_time_s"] / ok2["iters"]
ok2["log_tpi"] = np.log10(ok2["time_per_iter"].clip(lower=1e-9))
sc6 = ax6.scatter(ok2["total_size"], ok2["time_per_iter"], s=S, c=np.log10(ok2["n"].clip(1)),
                  cmap="viridis", alpha=ALPHA, zorder=3)
slope6, intercept6, r6, *_ = stats.linregress(ok2["log_size"], ok2["log_tpi"])
xfit6 = np.logspace(ok2["log_size"].min(), ok2["log_size"].max(), 200)
ax6.plot(xfit6, 10**intercept6 * xfit6**slope6, "k--", lw=1.2,
         label=f"fit: $t/k \\propto N^{{{slope6:.2f}}}$  ($R^2$={r6**2:.2f})")
plt.colorbar(sc6, ax=ax6, label="log₁₀(n)")
ax6.set_xscale("log"); ax6.set_yscale("log")
ax6.set_xlabel("N = n + m + p"); ax6.set_ylabel("Time per iteration (s)")
ax6.set_title("Per-iteration cost vs total size")
ax6.legend(fontsize=8); ax6.grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig("solver_scaling.png", dpi=150, bbox_inches="tight")
print("Saved solver_scaling.png")

# ── print regression summary ──────────────────────────────────────────────────
print("\n=== Power-law fit summary (log-log OLS on converged problems) ===")
print(f"  t ∝ n^{slope1:.3f}   R²={r1**2:.3f}  (n only)")
if len(m_ok) > 2:
    print(f"  t ∝ m^{slope2:.3f}   R²={r2**2:.3f}  (m only, m>0)")
if len(p_ok) > 2:
    print(f"  t ∝ p^{slope3:.3f}   R²={r3**2:.3f}  (p only, p>0)")
print(f"  t ∝ N^{slope4:.3f}   R²={r4**2:.3f}  (N = n+m+p)")
print(f"  t/k ∝ N^{slope6:.3f}  R²={r6**2:.3f}  (per-iteration cost vs N)")
