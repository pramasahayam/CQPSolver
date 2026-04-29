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

    // Cached transposes — avoid recomputing G^T and A^T every iteration.
    SpMat Gt_, At_;

    // Pre-allocated KKT matrix with precomputed value-index maps for O(nnz) updates.
    struct LhsCache {
        SpMat mat;
        bool  for_dual_reg = false;
        bool  valid        = false;
        std::vector<Eigen::Index> Q_offsets;     // offsets into mat.valuePtr() for each Q nnz
        std::vector<Eigen::Index> GtDG_offsets;  // offsets for each GtDG nnz (from first-iter GtDG pattern)
        std::vector<Eigen::Index> diag11_offsets; // offsets for the n diagonal entries of block (1,1)
        std::vector<Eigen::Index> diag22_offsets; // offsets for the m diagonal entries of block (2,2)
        std::vector<double>       Q_vals;         // fixed Q values (copied once)
    };
    LhsCache lhs_cache_;

    std::unique_ptr<UmfPackLUWithInfo> lhs_solver_;
    bool   needs_dual_reg_ = false;
    double current_delta_  = 0.0;

    // Find the compressed-storage offset of (row, col) in a CSC matrix.
    static Eigen::Index find_inner_offset(const SpMat& mat, int row, int col);

    // Build lhs_cache_.mat with the correct sparsity for dual_reg setting,
    // populate offset maps, and reset lhs_solver_.
    // GtDG must be the actual G^T*D*G for this iteration — its structural
    // non-zeros are used as the authoritative pattern (avoids bugs from
    // GtG pruning integer-cancellation zeros that GtDG later fills in).
    void init_lhs_cache(const SpMat& GtDG, bool dual_reg);

    // Update lhs_cache_.mat values in-place given new GtDG and delta.
    // Requires lhs_cache_.valid and lhs_cache_.for_dual_reg matching needs_dual_reg_.
    void update_lhs_in_place(const SpMat& GtDG, double delta);

    SpMat   build_lhs(const SpMat& lhs_11, bool dual_reg) const;
    void    factorize_lhs(const SpMat& lhs);

    void print_header() const;
    void print_row(const SolverState& state) const;
};
