#include "cqpsolver.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

// ─── Sparse helpers ──────────────────────────────────────────────────────────

static SpMat speye(int n, double scale = 1.0) {
    SpMat m(n, n);
    m.reserve(Eigen::VectorXi::Constant(n, 1));
    for (int i = 0; i < n; ++i) m.insert(i, i) = scale;
    m.makeCompressed();
    return m;
}

static SpMat vstack(const SpMat& A, const SpMat& B) {
    assert(A.cols() == B.cols());
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(A.nonZeros() + B.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k)
        for (SpMat::InnerIterator it(A, k); it; ++it)
            t.emplace_back(it.row(), it.col(), it.value());
    int offset = A.rows();
    for (int k = 0; k < B.outerSize(); ++k)
        for (SpMat::InnerIterator it(B, k); it; ++it)
            t.emplace_back(it.row() + offset, it.col(), it.value());
    SpMat result(A.rows() + B.rows(), A.cols());
    result.setFromTriplets(t.begin(), t.end());
    return result;
}

static SpMat hstack(const SpMat& A, const SpMat& B) {
    assert(A.rows() == B.rows());
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(A.nonZeros() + B.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k)
        for (SpMat::InnerIterator it(A, k); it; ++it)
            t.emplace_back(it.row(), it.col(), it.value());
    int offset = A.cols();
    for (int k = 0; k < B.outerSize(); ++k)
        for (SpMat::InnerIterator it(B, k); it; ++it)
            t.emplace_back(it.row(), it.col() + offset, it.value());
    SpMat result(A.rows(), A.cols() + B.cols());
    result.setFromTriplets(t.begin(), t.end());
    return result;
}

// [[A, B], [C, D]]
static SpMat block2x2(const SpMat& A, const SpMat& B,
                      const SpMat& C, const SpMat& D) {
    return vstack(hstack(A, B), hstack(C, D));
}

static SpMat row_subset(const SpMat& M, const std::vector<int>& rows) {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(rows.size() * (M.nonZeros() / std::max(Eigen::Index(1), M.rows())));
    for (int new_row = 0; new_row < static_cast<int>(rows.size()); ++new_row) {
        int old_row = rows[new_row];
        for (int k = 0; k < M.outerSize(); ++k)
            for (SpMat::InnerIterator it(M, k); it; ++it)
                if (it.row() == old_row)
                    t.emplace_back(new_row, it.col(), it.value());
    }
    SpMat result(static_cast<int>(rows.size()), M.cols());
    result.setFromTriplets(t.begin(), t.end());
    return result;
}

// ─── Print helpers ───────────────────────────────────────────────────────────

static const char* kDivider =
    "──────────────────────────────────────────────────────────────────────────────────────────────────────";

static const char* kHeader =
    "Iter. │   Objective    │  Primal Inequality  │  Primal Equality   │  Stationarity  │   Duality   │ Step Size";

// ─── Solver constructor ───────────────────────────────────────────────────────

Solver::Solver(Problem prob, double tol, int max_iter, bool quiet,
               double reg, int n_refine)
    : tol(tol), max_iter(max_iter), quiet(quiet), reg(reg), n_refine(n_refine),
      prob_(std::move(prob)) {}

// ─── build_lhs ───────────────────────────────────────────────────────────────

SpMat Solver::build_lhs(const SpMat& lhs_11, bool dual_reg) const {
    int m = prob_.m();
    if (m == 0) return lhs_11;
    SpMat At = prob_.A.transpose();
    if (dual_reg) {
        SpMat D22 = speye(m, -current_delta_);
        return block2x2(lhs_11, At, prob_.A, D22);
    }
    SpMat zero_mm(m, m);
    return block2x2(lhs_11, At, prob_.A, zero_mm);
}

// ─── factorize_lhs ───────────────────────────────────────────────────────────

void Solver::factorize_lhs(const SpMat& lhs) {
    if (lhs_solver_ == nullptr) {
        lhs_solver_ = std::make_unique<UmfPackLUWithInfo>();
        lhs_solver_->umfpackControl()[UMFPACK_PRL] = 0;
        lhs_solver_->compute(lhs);
    } else {
        lhs_solver_->factorize(lhs);
    }
    if (lhs_solver_->info() == Eigen::NumericalIssue && lhs_solver_->umfpackFactorizeReturncode() != UMFPACK_WARNING_singular_matrix) {
        throw std::runtime_error("UmfPackLU factorization failed.");
    }
}

// ─── solve_reduced_impl ──────────────────────────────────────────────────────
// The Python version captures z, s, prob from enclosing scope. In C++ we pass them.
static Solver::Direction solve_reduced_impl(
    const UmfPackLUWithInfo& solver,
    const SpMat& lhs,
    const SpMat& G,
    const Vec& z, const Vec& s,
    const Vec& r1, const Vec& r2, const Vec& r3, const Vec& r4,
    int n, int m,
    bool needs_dual_reg, double current_delta,
    int n_refine)
{
    Vec rhs_x = r1 - G.transpose() * ((r2 - z.cwiseProduct(r3)).cwiseQuotient(s));
    Vec rhs = (m > 0) ? (Vec(n + m) << rhs_x, r4).finished() : rhs_x;

    Vec sol = solver.solve(rhs);

    for (int iter = 0; iter < n_refine; ++iter) {
        Vec residual = rhs - lhs * sol;
        if (!needs_dual_reg) {
            if (m > 0)
                residual.head(n) += current_delta * sol.head(n);
            else
                residual += current_delta * sol;
        }
        sol += solver.solve(residual);
    }

    Vec dx = sol.head(n);
    Vec dy = (m > 0) ? Vec(sol.tail(m)) : Vec::Zero(0);
    Vec ds = r3 - G * dx;
    Vec dz = (r2 - z.cwiseProduct(ds)).cwiseQuotient(s);

    return {dx, ds, dz, dy};
}

// ─── find_initial_state ───────────────────────────────────────────────────────

SolverState Solver::find_initial_state() {
    int n = prob_.n(), m = prob_.m(), p = prob_.p();

    // Build (n+p+m) × (n+p+m) block system
    SpMat Q_reg = prob_.Q + speye(n, reg);
    SpMat top   = hstack(hstack(Q_reg, prob_.G.transpose()), prob_.A.transpose());

    SpMat mid_left  = hstack(prob_.G, speye(p, -1.0));
    SpMat mid_right(p, m);  // zero block
    SpMat mid = hstack(mid_left, mid_right);

    SpMat LHS;
    Vec   RHS;
    if (m > 0) {
        SpMat bot_left(m, n + p);
        SpMat bot_right = speye(m, -reg);
        SpMat bot = hstack(bot_left, bot_right);
        LHS = vstack(vstack(top, mid), bot);
        RHS = (Vec(n + p + m) << -prob_.q, prob_.h, prob_.b).finished();
    } else {
        LHS = vstack(top, mid);
        RHS = (Vec(n + p) << -prob_.q, prob_.h).finished();
    }

    UmfPackLUWithInfo init_solver;
    init_solver.umfpackControl()[UMFPACK_PRL] = 0;
    init_solver.compute(LHS);
    if (init_solver.info() != Eigen::Success)
        throw std::runtime_error("find_initial_state: failed to factorize initial KKT system.");

    Vec sol = init_solver.solve(RHS);
    Vec x   = sol.head(n);
    Vec y   = (m > 0) ? Vec(sol.tail(m)) : Vec::Zero(0);

    Vec z_raw = prob_.G * x - prob_.h;

    double alpha_p = (p > 0) ? z_raw.maxCoeff() : -1.0;
    Vec s0 = (alpha_p < 0) ? (-z_raw).eval() : (-z_raw).array() + 1.0 + alpha_p;

    double alpha_d = (p > 0) ? (-z_raw).maxCoeff() : -1.0;
    Vec z0 = (alpha_d < 0) ? z_raw.eval() : z_raw.array() + 1.0 + alpha_d;

    double obj = 0.5 * x.dot(prob_.Q * x) + prob_.q.dot(x);
    Residuals res = calc_residuals(x, s0, z0, y);

    SolverState state{0, obj, x, s0, z0, y, res, std::nullopt};

    if (!quiet) {
        print_header();
        print_row(state);
    }
    return state;
}

// ─── step ────────────────────────────────────────────────────────────────────

SolverState Solver::step(const SolverState& state) {
    int n = prob_.n(), m = prob_.m(), p = prob_.p();
    const Vec& s = state.s;
    const Vec& z = state.z;
    double delta = current_delta_;

    Vec zs = z.cwiseQuotient(s);

    // Reduced (n+m) × (n+m) LHS
    SpMat GTzsG = prob_.G.transpose() * zs.asDiagonal() * prob_.G;
    SpMat lhs_11 = prob_.Q + GTzsG + speye(n, delta);

    SpMat lhs = build_lhs(lhs_11, needs_dual_reg_);

    // Affine RHS
    Vec r1_aff = -(prob_.Q * state.x + prob_.q + prob_.G.transpose() * state.z + prob_.A.transpose() * state.y);
    Vec r2_aff = -s.cwiseProduct(z);
    Vec r3_aff = -(prob_.G * state.x + s - prob_.h);
    Vec r4_aff = -(prob_.A * state.x - prob_.b);

    // Factorize with cascading fallbacks
    constexpr double kMaxDelta = 1e-2;
    Direction aff;

    while (true) {
        bool singular_error = false;
        try {
            factorize_lhs(lhs);

            // Check for singular warning (only when dual reg not yet active)
            if (!needs_dual_reg_ && m > 0) {
                int retcode = lhs_solver_->umfpackFactorizeReturncode();
                if (retcode == UMFPACK_WARNING_singular_matrix) {
                    double rcond = lhs_solver_->rcond();
                    if (rcond < std::numeric_limits<double>::epsilon()) {
                        singular_error = true;
                    }
                }
            }

            if (!singular_error) {
                aff = solve_reduced_impl(*lhs_solver_, lhs, prob_.G, z, s,
                                        r1_aff, r2_aff, r3_aff, r4_aff,
                                        n, m, needs_dual_reg_, current_delta_, n_refine);
                break;
            }
        } catch (...) {
            singular_error = true;
        }

        if (singular_error && !needs_dual_reg_ && m > 0) {
            needs_dual_reg_ = true;
            lhs_solver_.reset();
            lhs = build_lhs(lhs_11, true);
        } else if (delta * 10.0 <= kMaxDelta) {
            delta *= 10.0;
            current_delta_ = delta;
            lhs_solver_.reset();
            lhs_11 = prob_.Q + GTzsG + speye(n, delta);
            lhs = build_lhs(lhs_11, needs_dual_reg_);
        } else {
            throw std::runtime_error("KKT system singular, all fallbacks exhausted.");
        }
    }

    // Centering-corrector (skipped when p=0)
    Direction cc;
    if (p > 0) {
        double mu    = s.dot(z) / p;
        double alpha = std::min({1.0, max_step(s, aff.ds), max_step(z, aff.dz)});
        Vec s_trial  = s + alpha * aff.ds;
        Vec z_trial  = z + alpha * aff.dz;
        double sigma = std::pow(s_trial.dot(z_trial) / s.dot(z), 3.0);
        Vec r2_cc    = Vec::Constant(p, sigma * mu) - aff.ds.cwiseProduct(aff.dz);
        cc = solve_reduced_impl(*lhs_solver_, lhs, prob_.G, z, s,
                                Vec::Zero(n), r2_cc, Vec::Zero(p), Vec::Zero(m),
                                n, m, needs_dual_reg_, current_delta_, n_refine);
    } else {
        cc = {Vec::Zero(n), Vec::Zero(p), Vec::Zero(p), Vec::Zero(m)};
    }

    Vec dx = aff.dx + cc.dx;
    Vec ds = aff.ds + cc.ds;
    Vec dz = aff.dz + cc.dz;
    Vec dy = aff.dy + cc.dy;

    double step_size = std::min(1.0, 0.99 * std::min(max_step(s, ds), max_step(z, dz)));

    Vec x_new = state.x + step_size * dx;
    Vec s_new = s       + step_size * ds;
    Vec z_new = z       + step_size * dz;
    Vec y_new = state.y + step_size * dy;

    double obj     = 0.5 * x_new.dot(prob_.Q * x_new) + prob_.q.dot(x_new);
    Residuals res  = calc_residuals(x_new, s_new, z_new, y_new);
    SolverState new_state{state.iter + 1, obj, x_new, s_new, z_new, y_new, res, step_size};

    if (!quiet) print_row(new_state);
    return new_state;
}

// ─── solve ───────────────────────────────────────────────────────────────────

std::pair<Result, std::vector<SolverState>> Solver::solve() {
    lhs_solver_.reset();
    needs_dual_reg_ = false;
    current_delta_  = reg;

    std::vector<SolverState> history;

    auto make_empty_state = [&]() {
        int n = prob_.n(), p = prob_.p(), m = prob_.m();
        return SolverState{0, 0.0, Vec::Zero(n), Vec::Zero(p), Vec::Zero(p),
                           Vec::Zero(m), {0.0, 0.0, 0.0, 0.0}, std::nullopt};
    };

    try {
        history.push_back(find_initial_state());
    } catch (const std::exception& e) {
        std::string msg = std::string("Failed to find initial state, error: ") + e.what();
        if (!quiet) std::printf("%s\n", msg.c_str());
        return {Result{false, msg, make_empty_state()}, {}};
    }

    try {
        while (history.back().iter < max_iter) {
            if (converged(history.back())) break;
            history.push_back(step(history.back()));
        }
    } catch (const std::exception& e) {
        std::string msg = std::string("Failed while solving, error: ") + e.what();
        if (!quiet) {
            std::printf("%s\n", kDivider);
            std::printf("%s\n", msg.c_str());
        }
        return {Result{false, msg, history.back()}, history};
    }

    SolverState final_state = history.back();
    bool conv = converged(final_state);
    std::string msg;
    if (conv) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "Solved in %d iterations, objective value = %.8g.",
                      final_state.iter, final_state.obj);
        msg = buf;
    } else {
        msg = "Failed to converge before iteration limit.";
    }

    if (!quiet) {
        std::printf("%s\n", kDivider);
        std::printf("%s\n", msg.c_str());
    }

    return {Result{conv, msg, final_state}, history};
}

// ─── calc_residuals ───────────────────────────────────────────────────────────

Residuals Solver::calc_residuals(const Vec& x, const Vec& s,
                                 const Vec& z, const Vec& y) const {
    double primal_ineq   = (prob_.G * x + s - prob_.h).lpNorm<Eigen::Infinity>();
    double primal_eq     = (prob_.A * x - prob_.b).lpNorm<Eigen::Infinity>();
    double stationarity  = (prob_.Q * x + prob_.q
                            + prob_.G.transpose() * z
                            + prob_.A.transpose() * y).lpNorm<Eigen::Infinity>();
    double duality       = std::abs(z.dot(s));
    return {primal_ineq, primal_eq, stationarity, duality};
}

// ─── max_step ────────────────────────────────────────────────────────────────

double Solver::max_step(const Vec& v, const Vec& dv) const {
    double result = std::numeric_limits<double>::infinity();
    for (int i = 0; i < dv.size(); ++i) {
        if (dv[i] < 0.0) result = std::min(result, -v[i] / dv[i]);
    }
    return result;
}

// ─── converged ───────────────────────────────────────────────────────────────

bool Solver::converged(const SolverState& state) const {
    const Residuals& res = state.residuals;

    double scale_primal = 1.0 + std::max({
        (prob_.A * state.x).lpNorm<Eigen::Infinity>(),
        prob_.b.lpNorm<Eigen::Infinity>(),
        (prob_.G * state.x).lpNorm<Eigen::Infinity>(),
        prob_.h.lpNorm<Eigen::Infinity>(),
        state.s.lpNorm<Eigen::Infinity>(),
    });
    bool primal_feas = std::max(res.primal_ineq, res.primal_eq) < tol * scale_primal;

    double scale_stat = 1.0 + std::max({
        (prob_.Q * state.x).lpNorm<Eigen::Infinity>(),
        (prob_.A.transpose() * state.y).lpNorm<Eigen::Infinity>(),
        (prob_.G.transpose() * state.z).lpNorm<Eigen::Infinity>(),
        prob_.q.lpNorm<Eigen::Infinity>(),
    });
    bool stat_check = res.stationarity < tol * scale_stat;

    double primal_obj = 0.5 * state.x.dot(prob_.Q * state.x) + prob_.q.dot(state.x);
    double dual_obj   = -0.5 * state.x.dot(prob_.Q * state.x)
                        - prob_.b.dot(state.y)
                        - prob_.h.dot(state.z);
    double scale_dual = 1.0 + std::max({1.0, std::abs(primal_obj), std::abs(dual_obj)});
    bool dual_check   = res.duality < tol * scale_dual;

    return primal_feas && stat_check && dual_check;
}

// ─── print helpers ───────────────────────────────────────────────────────────

void Solver::print_header() const {
    std::printf("%s\n%s\n%s\n", kDivider, kHeader, kDivider);
}

void Solver::print_row(const SolverState& state) const {
    const Residuals& res = state.residuals;
    char step_buf[16];
    if (state.step_size.has_value())
        std::snprintf(step_buf, sizeof(step_buf), "%9.4f", *state.step_size);
    else
        std::snprintf(step_buf, sizeof(step_buf), "    \xe2\x80\x94    ");  // UTF-8 em dash

    std::printf("%5d \xe2\x94\x82 %14.8g \xe2\x94\x82 %17.4e \xe2\x94\x82 %15.4e \xe2\x94\x82 %12.4e \xe2\x94\x82 %11.4e \xe2\x94\x82 %s\n",
                state.iter, state.obj,
                res.primal_ineq, res.primal_eq,
                res.stationarity, res.duality,
                step_buf);
}
