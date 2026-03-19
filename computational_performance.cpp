#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <iomanip>
#include <queue>
#include <algorithm>

#include "VectorDouble.hpp"
#include "DenseSquareMatrixDouble.hpp"
#include "SparseSquareMatrixCRSDouble.hpp"
#include "LinearSystemDense.hpp"
#include "LinearSystemSparse.hpp"

static VectorDouble random_vec(std::size_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    VectorDouble v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

static void sync_vector_reorder(VectorDouble& v, const std::vector<std::size_t>& p) {
    std::size_t n = v.size();
    VectorDouble temp(n);
    for (std::size_t i = 0; i < n; ++i) {
        temp[p[i]] = v[i];
    }
    v = std::move(temp);
}


template <class Fn>
static double time_per_call(std::size_t n, Fn&& fn) {
    if (n < 1000) {
        const int samples = 50;
        using clock = std::chrono::high_resolution_clock;
        auto t0 = clock::now();
        for (int k = 0; k < samples; ++k) fn();
        auto t1 = clock::now();
        return std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(samples);
    } else {
        using clock = std::chrono::high_resolution_clock;
        auto t0 = clock::now();
        fn();
        auto t1 = clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }
}


static inline std::size_t get_idx(std::size_t r, std::size_t c, std::size_t m) {
    return r * m + c;
}

static DenseSquareMatrixDouble build_laplace_dense(std::size_t m) {
    const std::size_t n = m * m;
    DenseSquareMatrixDouble A(n);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < m; ++c) {
            const std::size_t i = get_idx(r, c, m);
            double diag = 0.0;
            if (r > 0) { A(i, get_idx(r - 1, c, m)) = -1.0; diag += 1.0; }
            if (r + 1 < m) { A(i, get_idx(r + 1, c, m)) = -1.0; diag += 1.0; }
            if (c > 0) { A(i, get_idx(r, c - 1, m)) = -1.0; diag += 1.0; }
            if (c + 1 < m) { A(i, get_idx(r, c + 1, m)) = -1.0; diag += 1.0; }
            A(i, i) = diag + 0.5; 
        }
    }
    return A;
}

static SparseSquareMatrixCRSDouble build_laplace_sparse(std::size_t m) {
    const std::size_t n = m * m;
    SparseSquareMatrixCRSDouble A(n);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < m; ++c) {
            const std::size_t i = get_idx(r, c, m);
            double diag = 0.0;
            if (r > 0) { A.addEntry(i, get_idx(r - 1, c, m), -1.0); diag += 1.0; }
            if (r + 1 < m) { A.addEntry(i, get_idx(r + 1, c, m), -1.0); diag += 1.0; }
            if (c > 0) { A.addEntry(i, get_idx(r, c - 1, m), -1.0); diag += 1.0; }
            if (c + 1 < m) { A.addEntry(i, get_idx(r, c + 1, m), -1.0); diag += 1.0; }
            A.addEntry(i, i, diag + 0.5);
        }
    }
    A.finalize();
    return A;
}


int main() {
    std::vector<std::size_t> target_ns = {16, 64, 256, 1024, 4096, 10000, 22500, 40000, 50176};
    
    std::mt19937_64 rng(123456);
    std::ofstream out("bench_performance.csv");
    out << "operation,storage,n,execution_time_sec\n";

    volatile double sink = 0.0;

    std::cout << "Starting Comprehensive Benchmark..." << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (std::size_t n_target : target_ns) {
        std::size_t m = static_cast<std::size_t>(std::sqrt(n_target));
        std::size_t n = m * m;

        std::cout << ">>> Scaling: n = " << std::setw(6) << n << " (m = " << m << ")" << std::endl;

        VectorDouble x_orig = random_vec(n, rng);
        VectorDouble b_orig = random_vec(n, rng);

        SparseSquareMatrixCRSDouble A_std = build_laplace_sparse(m);
        SparseSquareMatrixCRSDouble A_std_copy = A_std;
        LinearSystemSparse sys_std{ std::move(A_std_copy), VectorDouble(x_orig), VectorDouble(b_orig) };

        out << "vec_add,sparse," << n << "," << time_per_call(n, [&](){ auto z = x_orig + b_orig; sink += z[0]; }) << "\n";
        out << "mat_add,sparse_standard," << n << "," << time_per_call(n, [&](){ auto C = sys_std.A() + A_std; sink += C.nnz(); }) << "\n";
        out << "matvec,sparse_standard," << n << "," << time_per_call(n, [&](){ auto y = sys_std.A() * sys_std.x(); sink += y[0]; }) << "\n";
        out << "res_nominal,sparse_standard," << n << "," << time_per_call(n, [&](){ auto r = sys_std.residual(); sink += r[0]; }) << "\n";
        out << "res_optimised,sparse_standard," << n << "," << time_per_call(n, [&](){ auto r = sys_std.residual_opt(); sink += r[0]; }) << "\n";

        std::vector<std::size_t> p = A_std.computeRCM();
        A_std.applyPermutation(p);
        VectorDouble x_rcm = x_orig;
        VectorDouble b_rcm = b_orig;
        sync_vector_reorder(x_rcm, p);
        sync_vector_reorder(b_rcm, p);
        LinearSystemSparse sys_rcm{ std::move(A_std), std::move(x_rcm), std::move(b_rcm) };

        out << "matvec,sparse_rcm," << n << "," << time_per_call(n, [&](){ auto y = sys_rcm.A() * sys_rcm.x(); sink += y[0]; }) << "\n";
        out << "res_optimised,sparse_rcm," << n << "," << time_per_call(n, [&](){ auto r = sys_rcm.residual_opt(); sink += r[0]; }) << "\n";

        if (n < 22500) {
            DenseSquareMatrixDouble A_dense = build_laplace_dense(m);
            DenseSquareMatrixDouble A_dense_2 = build_laplace_dense(m);
            LinearSystemDense sys_dense{ std::move(A_dense), VectorDouble(x_orig), VectorDouble(b_orig) };

            out << "mat_add,dense," << n << "," << time_per_call(n, [&](){ auto C = sys_dense.A() + A_dense_2; sink += C(0,0); }) << "\n";
            out << "matvec,dense," << n << "," << time_per_call(n, [&](){ auto y = sys_dense.A() * sys_dense.x(); sink += y[0]; }) << "\n";
            out << "res_nominal,dense," << n << "," << time_per_call(n, [&](){ auto r = sys_dense.residual(); sink += r[0]; }) << "\n";
        } else {
            std::cout << "    (Dense matrix skipped for n > 22500 due to O(n^2) memory)" << std::endl;
        }
    }

    out.close();
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Benchmark complete. Data exported to 'bench_performance.csv'." << std::endl;

    if (sink == 0.12345) std::cout << sink << std::endl;

    return 0;
}