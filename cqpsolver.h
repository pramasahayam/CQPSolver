#pragma once

#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>
#include <optional>
#include <string>
#include <vector>

using SpMat = Eigen::SparseMatrix<double>;
using Vec   = Eigen::VectorXd;

// Exposes the protected m_umfpackInfo array so we can read rcond after factorize().
class UmfPackLUWithInfo : public Eigen::UmfPackLU<SpMat> {
public:
    double rcond() const { return m_umfpackInfo[UMFPACK_RCOND]; }
};

struct Problem {
    SpMat Q;  // (n×n)
    Vec   q;  // (n,)
    SpMat G;  // (p×n)
    Vec   h;  // (p,)
    SpMat A;  // (m×n)
    Vec   b;  // (m,)

    int n() const { return static_cast<int>(q.size()); }
    int p() const { return static_cast<int>(h.size()); }
    int m() const { return static_cast<int>(b.size()); }
};

struct Residuals {
    double primal_ineq;
    double primal_eq;
    double stationarity;
    double duality;
};

struct SolverState {
    int    iter;
    double obj;
    Vec    x, s, z, y;
    Residuals             residuals;
    std::optional<double> step_size;
};

struct Result {
    bool        convergence;
    std::string msg;
    SolverState final_state;
};

class Solver {
public:
    double tol      = 1e-8;
    int    max_iter = 25;
    bool   quiet    = false;
    double reg      = 1e-8;
    int    n_refine = 0;

    explicit Solver(Problem prob,
                    double tol      = 1e-8,
                    int    max_iter = 25,
                    bool   quiet    = false,
                    double reg      = 1e-8,
                    int    n_refine = 0);

    SolverState find_initial_state();
    SolverState step(const SolverState& state);
    std::pair<Result, std::vector<SolverState>> solve();

    Residuals calc_residuals(const Vec& x, const Vec& s,
                             const Vec& z, const Vec& y) const;
    bool      converged(const SolverState& state) const;
    double    max_step(const Vec& v, const Vec& dv) const;

    struct Direction { Vec dx, ds, dz, dy; };

private:
    Problem prob_;

    std::unique_ptr<UmfPackLUWithInfo> lhs_solver_;
    bool   needs_dual_reg_ = false;
    double current_delta_  = 0.0;

    SpMat   build_lhs(const SpMat& lhs_11, bool dual_reg) const;
    void    factorize_lhs(const SpMat& lhs);

    void print_header() const;
    void print_row(const SolverState& state) const;
};
