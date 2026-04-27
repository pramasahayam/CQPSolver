#include "cqpsolver.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <matio.h>

namespace fs = std::filesystem;

static const fs::path kMatDir     = "../QP-Test-Problems/MAT_Files";
static const fs::path kOutputCsv  = "mm_benchmark_results_cpp.csv";
static const std::set<std::string> kSkip = {"BOYD2"};

// ─── matio helpers ───────────────────────────────────────────────────────────

static Vec read_vec(matvar_t* var) {
    assert(var != nullptr);
    size_t len = var->dims[0] * (var->rank > 1 ? var->dims[1] : 1);
    Vec v(static_cast<int>(len));
    switch (var->data_type) {
        case MAT_T_DOUBLE: {
            auto* d = static_cast<double*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = d[i];
            break;
        }
        case MAT_T_SINGLE: {
            auto* d = static_cast<float*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = static_cast<double>(d[i]);
            break;
        }
        case MAT_T_INT8: {
            auto* d = static_cast<int8_t*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = static_cast<double>(d[i]);
            break;
        }
        case MAT_T_UINT8: {
            auto* d = static_cast<uint8_t*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = static_cast<double>(d[i]);
            break;
        }
        case MAT_T_INT16: {
            auto* d = static_cast<int16_t*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = static_cast<double>(d[i]);
            break;
        }
        case MAT_T_UINT16: {
            auto* d = static_cast<uint16_t*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = static_cast<double>(d[i]);
            break;
        }
        case MAT_T_INT32: {
            auto* d = static_cast<int32_t*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = static_cast<double>(d[i]);
            break;
        }
        case MAT_T_UINT32: {
            auto* d = static_cast<uint32_t*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = static_cast<double>(d[i]);
            break;
        }
        case MAT_T_INT64: {
            auto* d = static_cast<int64_t*>(var->data);
            for (int i = 0; i < (int)len; ++i) v[i] = static_cast<double>(d[i]);
            break;
        }
        default:
            throw std::runtime_error("read_vec: unsupported data type " +
                                     std::to_string(var->data_type));
    }
    return v;
}

static SpMat read_sparse(matvar_t* var) {
    assert(var != nullptr && var->class_type == MAT_C_SPARSE);
    int nrows = static_cast<int>(var->dims[0]);
    int ncols = static_cast<int>(var->dims[1]);
    auto* sp  = static_cast<mat_sparse_t*>(var->data);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(sp->ndata);
    for (int col = 0; col < ncols; ++col) {
        for (int idx = sp->jc[col]; idx < sp->jc[col + 1]; ++idx) {
            triplets.emplace_back(sp->ir[idx], col,
                                  static_cast<double*>(sp->data)[idx]);
        }
    }
    SpMat mat(nrows, ncols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

// ─── vstack / row_subset (duplicated from cqpsolver.cpp for benchmark use) ───

static SpMat sp_vstack(const SpMat& A, const SpMat& B) {
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
    SpMat r(A.rows() + B.rows(), A.cols());
    r.setFromTriplets(t.begin(), t.end());
    return r;
}

static SpMat sp_row_subset(const SpMat& M, const std::vector<int>& rows) {
    // Build a reverse index for efficiency
    std::vector<int> new_row_idx(M.rows(), -1);
    for (int i = 0; i < (int)rows.size(); ++i) new_row_idx[rows[i]] = i;

    std::vector<Eigen::Triplet<double>> t;
    t.reserve(M.nonZeros());
    for (int k = 0; k < M.outerSize(); ++k)
        for (SpMat::InnerIterator it(M, k); it; ++it)
            if (new_row_idx[it.row()] >= 0)
                t.emplace_back(new_row_idx[it.row()], it.col(), it.value());
    SpMat r(static_cast<int>(rows.size()), M.cols());
    r.setFromTriplets(t.begin(), t.end());
    return r;
}

static SpMat sp_speye(int n, double scale = 1.0) {
    SpMat m(n, n);
    m.reserve(Eigen::VectorXi::Constant(n, 1));
    for (int i = 0; i < n; ++i) m.insert(i, i) = scale;
    m.makeCompressed();
    return m;
}

// ─── parse_mat ───────────────────────────────────────────────────────────────

static Problem parse_mat(const fs::path& filepath) {
    mat_t* mat = Mat_Open(filepath.string().c_str(), MAT_ACC_RDONLY);
    if (!mat)
        throw std::runtime_error("Failed to open: " + filepath.string());

    auto read_var = [&](const char* name) {
        matvar_t* v = Mat_VarRead(mat, name);
        if (!v) throw std::runtime_error(std::string("Variable not found: ") + name);
        return v;
    };

    matvar_t* vQ  = read_var("Q");
    matvar_t* vq  = read_var("c");
    matvar_t* vA  = read_var("A");
    matvar_t* vrl = read_var("rl");
    matvar_t* vru = read_var("ru");
    matvar_t* vlb = read_var("lb");
    matvar_t* vub = read_var("ub");

    SpMat Q = read_sparse(vQ);
    Vec   q = read_vec(vq);
    SpMat A = read_sparse(vA);
    Vec   rl = read_vec(vrl);
    Vec   ru = read_vec(vru);
    Vec   lb = read_vec(vlb);
    Vec   ub = read_vec(vub);

    Mat_VarFree(vQ); Mat_VarFree(vq); Mat_VarFree(vA);
    Mat_VarFree(vrl); Mat_VarFree(vru); Mat_VarFree(vlb); Mat_VarFree(vub);
    Mat_Close(mat);

    int n = Q.rows();
    int num_constraints = static_cast<int>(rl.size());

    // Equality mask: rl == ru
    std::vector<int> eq_idx, ineq_idx;
    for (int i = 0; i < num_constraints; ++i) {
        if (rl[i] == ru[i]) eq_idx.push_back(i);
        else                 ineq_idx.push_back(i);
    }

    // A_eq, b_eq
    SpMat A_eq(0, n), A_ineq(0, n);
    Vec   b_eq = Vec::Zero(0);
    Vec   ru_eq = Vec::Zero(0), rl_ineq = Vec::Zero(0), ru_ineq = Vec::Zero(0);

    if (!eq_idx.empty()) {
        A_eq  = sp_row_subset(A, eq_idx);
        b_eq.resize(eq_idx.size());
        for (int i = 0; i < (int)eq_idx.size(); ++i) b_eq[i] = ru[eq_idx[i]];
    }

    if (!ineq_idx.empty()) {
        A_ineq.resize(ineq_idx.size(), n);
        A_ineq = sp_row_subset(A, ineq_idx);
        rl_ineq.resize(ineq_idx.size());
        ru_ineq.resize(ineq_idx.size());
        for (int i = 0; i < (int)ineq_idx.size(); ++i) {
            rl_ineq[i] = rl[ineq_idx[i]];
            ru_ineq[i] = ru[ineq_idx[i]];
        }
    }

    // G_ineq = [A_ineq; -A_ineq], h_ineq = [ru_ineq; -rl_ineq]
    SpMat G_ineq(0, n);
    Vec   h_ineq = Vec::Zero(0);
    if (!ineq_idx.empty()) {
        SpMat neg_A_ineq = -A_ineq;
        G_ineq = sp_vstack(A_ineq, neg_A_ineq);
        h_ineq.resize(2 * ineq_idx.size());
        h_ineq << ru_ineq, -rl_ineq;
    }

    // G_full = [G_ineq; I_n; -I_n], h_full = [h_ineq; ub; -lb]
    SpMat eye_n   = sp_speye(n,  1.0);
    SpMat neye_n  = sp_speye(n, -1.0);
    SpMat G_full  = sp_vstack(sp_vstack(G_ineq, eye_n), neye_n);
    Vec   h_full(G_full.rows());
    h_full << h_ineq, ub, -lb;

    // Finite mask
    std::vector<int> finite_rows;
    for (int i = 0; i < h_full.size(); ++i)
        if (std::isfinite(h_full[i])) finite_rows.push_back(i);

    SpMat G = sp_row_subset(G_full, finite_rows);
    Vec   h(finite_rows.size());
    for (int i = 0; i < (int)finite_rows.size(); ++i) h[i] = h_full[finite_rows[i]];

    return Problem{Q, q, G, h, A_eq, b_eq};
}

// ─── CSV helpers ─────────────────────────────────────────────────────────────

// Quote a field if it contains a comma, quote, or newline (per RFC 4180).
static std::string csv_quote(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else          out += c;
    }
    out += '"';
    return out;
}

// ─── solve_problem ───────────────────────────────────────────────────────────

struct BenchRow {
    std::string name;
    int n = 0, m = 0, p = 0;
    bool converged = false;
    int  iters = 0;
    double obj = std::numeric_limits<double>::quiet_NaN();
    double solve_time_s = std::numeric_limits<double>::quiet_NaN();
    double primal_ineq = std::numeric_limits<double>::quiet_NaN();
    double primal_eq   = std::numeric_limits<double>::quiet_NaN();
    double stationarity = std::numeric_limits<double>::quiet_NaN();
    double duality = std::numeric_limits<double>::quiet_NaN();
    std::string step_size_str;
    std::string msg;
    std::string error;
};

static BenchRow solve_problem(const std::string& name, const fs::path& filepath) {
    BenchRow row;
    row.name = name;
    try {
        Problem prob = parse_mat(filepath);
        row.n = prob.n();
        row.m = prob.m();
        row.p = prob.p();

        Solver solver(std::move(prob), 1e-8, 100, /*quiet=*/true, 1e-8, /*n_refine=*/2);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto [result, _] = solver.solve();
        auto t1 = std::chrono::high_resolution_clock::now();
        row.solve_time_s = std::chrono::duration<double>(t1 - t0).count();

        const SolverState& fs = result.final_state;
        row.converged    = result.convergence;
        row.iters        = fs.iter;
        row.obj          = fs.obj;
        row.primal_ineq  = fs.residuals.primal_ineq;
        row.primal_eq    = fs.residuals.primal_eq;
        row.stationarity = fs.residuals.stationarity;
        row.duality      = fs.residuals.duality;
        row.step_size_str = fs.step_size.has_value()
                            ? std::to_string(*fs.step_size) : "";
        row.msg = result.msg;
    } catch (const std::exception& e) {
        row.error = std::string(typeid(e).name()) + ": " + e.what();
        row.msg   = "ERROR";
    }
    return row;
}

// ─── live progress ───────────────────────────────────────────────────────────

static void print_live_row(int idx, int total, const BenchRow& row) {
    if (!row.error.empty()) {
        std::printf("%4d/%-4d  %-14s  ERR   %s\n",
                    idx, total, row.name.c_str(), row.error.c_str());
        std::fflush(stdout);
        return;
    }
    const char* status = row.converged ? "OK  " : "FAIL";
    std::printf("%4d/%-4d %-14s  n=%5d  m=%5d  p=%6d  %s  %4d iters  %8.3fs  obj= %14.6e\n",
                idx, total, row.name.c_str(),
                row.n, row.m, row.p,
                status, row.iters, row.solve_time_s, row.obj);
    std::fflush(stdout);
}

// ─── write CSV row ────────────────────────────────────────────────────────────

static void write_csv_row(std::ofstream& f, const BenchRow& row) {
    auto nan_str = [](double v) {
        return std::isnan(v) ? std::string("") : std::to_string(v);
    };
    // Use sufficient precision for obj and residuals
    auto sci = [](double v, int prec = 15) -> std::string {
        if (std::isnan(v)) return "";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*e", prec, v);
        return buf;
    };

    f << csv_quote(row.name) << ","
      << (row.n ? std::to_string(row.n) : "") << ","
      << (row.m >= 0 && !std::isnan(row.obj) ? std::to_string(row.m) : "") << ","
      << (row.p >= 0 && !std::isnan(row.obj) ? std::to_string(row.p) : "") << ","
      << (row.error.empty() ? (row.converged ? "True" : "False") : "False") << ","
      << (row.iters >= 0 && !std::isnan(row.obj) ? std::to_string(row.iters) : "") << ","
      << sci(row.obj) << ","
      << sci(row.solve_time_s) << ","
      << sci(row.primal_ineq) << ","
      << sci(row.primal_eq) << ","
      << sci(row.stationarity) << ","
      << sci(row.duality) << ","
      << csv_quote(row.step_size_str) << ","
      << csv_quote(row.msg) << ","
      << csv_quote(row.error) << "\n";
}

// ─── run_benchmark ───────────────────────────────────────────────────────────

static void run_benchmark(const fs::path& mat_dir, const fs::path& output_csv,
                          const std::set<std::string>& skip,
                          const std::vector<std::string>& problems)
{
    std::vector<fs::path> mat_files;
    if (!problems.empty()) {
        for (const auto& name : problems) {
            fs::path p = mat_dir / (name + ".mat");
            if (fs::exists(p)) mat_files.push_back(p);
            else std::fprintf(stderr, "Warning: file not found: %s\n", p.string().c_str());
        }
    } else {
        for (const auto& entry : fs::directory_iterator(mat_dir)) {
            if (entry.path().extension() != ".mat") continue;
            if (skip.count(entry.path().stem().string())) continue;
            mat_files.push_back(entry.path());
        }
        std::sort(mat_files.begin(), mat_files.end());
    }

    int total = static_cast<int>(mat_files.size());
    std::printf("Running %d problems  →  %s\n", total, output_csv.string().c_str());
    if (!skip.empty() && problems.empty())
        std::printf("Skipping: BOYD2\n");
    std::printf("\n");

    std::ofstream csv(output_csv);
    csv << "name,n,m,p,converged,iters,obj,solve_time_s,"
           "primal_ineq,primal_eq,stationarity,duality,step_size,msg,error\n";

    int solved = 0, failed = 0;
    for (int i = 0; i < total; ++i) {
        BenchRow row = solve_problem(mat_files[i].stem().string(), mat_files[i]);
        print_live_row(i + 1, total, row);
        write_csv_row(csv, row);
        csv.flush();

        if (!row.error.empty() || !row.converged) ++failed;
        else                                        ++solved;
    }

    std::printf("\nSolved %d/%d  •  Failed/ERR %d/%d  •  Results in %s\n",
                solved, total, failed, total, output_csv.string().c_str());
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::vector<std::string> problems;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--problems") == 0 && i + 1 < argc) {
            std::istringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                // trim whitespace
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                if (!token.empty()) problems.push_back(token);
            }
        }
    }

    run_benchmark(kMatDir, kOutputCsv, kSkip, problems);
    return 0;
}
