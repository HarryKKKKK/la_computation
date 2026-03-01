#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

#include "VectorDouble.hpp"
#include "DenseSquareMatrixDouble.hpp"
#include "SparseSquareMatrixCRSDouble.hpp"
#include "LinearSystemDense.hpp"
#include "LinearSystemSparse.hpp"

static const int GLOBAL_REPEATS = 10;

// ======================================================
// timing helpers
// ======================================================
template <class Fn>
static double time_once_seconds(Fn&& fn)
{
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    fn();
    auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

template <class Fn>
static double time_many_seconds(int inner, Fn&& fn)
{
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    for (int k = 0; k < inner; ++k) fn();
    auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// N < 5000: do m=10 repeats and average
// N >= 5000: do 1 repeat
template <class Fn>
static double time_per_call(std::size_t N, Fn&& fn)
{
    if (N < 5000) {
        const int m = 10;
        return time_many_seconds(m, std::forward<Fn>(fn)) / static_cast<double>(m);
    } else {
        return time_once_seconds(std::forward<Fn>(fn));
    }
}

// ======================================================
// random generators
// ======================================================
static double rnd(std::mt19937_64& rng)
{
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

static VectorDouble random_vec(std::size_t N, std::mt19937_64& rng)
{
    VectorDouble v(N);
    for (std::size_t i = 0; i < N; ++i)
        v[i] = rnd(rng);
    return v;
}

static DenseSquareMatrixDouble random_dense(std::size_t N, std::mt19937_64& rng)
{
    DenseSquareMatrixDouble A(N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            A(i, j) = rnd(rng);
    return A;
}

// sparse with probability p_nz for off-diagonal entries
static SparseSquareMatrixCRSDouble random_sparse(std::size_t N,
                                                 std::mt19937_64& rng,
                                                 double p_nz)
{
    SparseSquareMatrixCRSDouble A(N);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (std::size_t i = 0; i < N; ++i) {
        A.addEntry(i, i, 5.0 + std::abs(rnd(rng))); // strong diagonal
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            if (coin(rng) < p_nz)
                A.addEntry(i, j, rnd(rng));
        }
    }

    A.finalize();
    return A;
}

// ======================================================
// CSV writer
// ======================================================
static void write_row(std::ofstream& out,
                      const std::string& op,
                      const std::string& storage,
                      std::size_t N,
                      double avg_seconds)
{
    out << op << "," << storage << "," << N << "," << avg_seconds << "\n";
}

int main()
{
    std::vector<std::size_t> Ns = {10, 100, 500, 1000, 5000, 10000};

    const double p_nz = 0.5;
    std::mt19937_64 rng(123456);

    std::ofstream out("bench_random.csv");
    out << "op,storage,N,time_per_call\n";

    volatile double sink = 0.0;

    for (std::size_t N : Ns)
    {
        std::cout << "Benchmark N = " << N
                  << " (GLOBAL_REPEATS=" << GLOBAL_REPEATS << ")\n";

        // accumulators (sum of per-call times across GLOBAL_REPEATS)
        double t_vec_add = 0.0, t_vec_sub = 0.0, t_vec_scalar = 0.0;
        double t_vec_n1  = 0.0, t_vec_n2  = 0.0, t_vec_ninf  = 0.0;

        double t_sysD_scalar = 0.0, t_sysD_diag = 0.0, t_sysD_symm = 0.0;
        double t_sysD_matvec = 0.0, t_sysD_res  = 0.0;

        double t_sysS_scalar = 0.0, t_sysS_diag = 0.0, t_sysS_symm = 0.0;
        double t_sysS_matvec = 0.0, t_sysS_res  = 0.0;

        const double a_vec = 2.3;
        const double a_sys = 1.7;

        for (int r = 0; r < GLOBAL_REPEATS; ++r)
        {
            // Re-generate new random data every repeat (good)
            VectorDouble x = random_vec(N, rng);
            VectorDouble y = random_vec(N, rng);

            DenseSquareMatrixDouble A_dense = random_dense(N, rng);
            SparseSquareMatrixCRSDouble A_sparse = random_sparse(N, rng, p_nz);

            VectorDouble b_dense  = random_vec(N, rng);
            VectorDouble b_sparse = random_vec(N, rng);

            LinearSystemDense sysD{
                DenseSquareMatrixDouble(A_dense),
                VectorDouble(x),
                VectorDouble(b_dense)
            };

            LinearSystemSparse sysS{
                SparseSquareMatrixCRSDouble(A_sparse),
                VectorDouble(x),
                VectorDouble(b_sparse)
            };

            // VECTOR ops
            t_vec_add += time_per_call(N, [&](){ auto z = x + y; sink += z[0]; });
            t_vec_sub += time_per_call(N, [&](){ auto z = x - y; sink += z[0]; });
            t_vec_scalar += time_per_call(N, [&](){ auto z = x * a_vec; sink += z[0]; });

            t_vec_n1 += time_per_call(N, [&](){ sink += x.norm_n(1); });
            t_vec_n2 += time_per_call(N, [&](){ sink += x.norm_n(2); });
            t_vec_ninf += time_per_call(N, [&](){ sink += x.normInf(); });

            // DENSE system ops
            t_sysD_scalar += time_per_call(N, [&](){ auto s = sysD * a_sys; sink += s.b()[0]; });
            t_sysD_diag   += time_per_call(N, [&](){ sink += sysD.isDiagonallyDominant() ? 1.0 : 0.0; });
            t_sysD_symm   += time_per_call(N, [&](){ sink += sysD.isSymmetric() ? 1.0 : 0.0; });
            t_sysD_matvec += time_per_call(N, [&](){ auto yv = sysD.A() * sysD.x(); sink += yv[0]; });
            t_sysD_res    += time_per_call(N, [&](){ auto rv = sysD.residual(); sink += rv[0]; });

            // SPARSE system ops
            t_sysS_scalar += time_per_call(N, [&](){ auto s = sysS * a_sys; sink += s.b()[0]; });
            t_sysS_diag   += time_per_call(N, [&](){ sink += sysS.isDiagonallyDominant() ? 1.0 : 0.0; });
            t_sysS_symm   += time_per_call(N, [&](){ sink += sysS.isSymmetric() ? 1.0 : 0.0; });
            t_sysS_matvec += time_per_call(N, [&](){ auto yv = sysS.A() * sysS.x(); sink += yv[0]; });
            t_sysS_res    += time_per_call(N, [&](){ auto rv = sysS.residual(); sink += rv[0]; });
        }

        const double denom = static_cast<double>(GLOBAL_REPEATS);

        // write averages over GLOBAL_REPEATS
        write_row(out, "vec_add",     "na",     N, t_vec_add / denom);
        write_row(out, "vec_sub",     "na",     N, t_vec_sub / denom);
        write_row(out, "vec_scalar",  "na",     N, t_vec_scalar / denom);
        write_row(out, "vec_norm1",   "na",     N, t_vec_n1 / denom);
        write_row(out, "vec_norm2",   "na",     N, t_vec_n2 / denom);
        write_row(out, "vec_normInf", "na",     N, t_vec_ninf / denom);

        write_row(out, "sys_scalar",  "dense",  N, t_sysD_scalar / denom);
        write_row(out, "diag_dom",    "dense",  N, t_sysD_diag / denom);
        write_row(out, "symm",        "dense",  N, t_sysD_symm / denom);
        write_row(out, "matvec",      "dense",  N, t_sysD_matvec / denom);
        write_row(out, "residual",    "dense",  N, t_sysD_res / denom);

        write_row(out, "sys_scalar",  "sparse", N, t_sysS_scalar / denom);
        write_row(out, "diag_dom",    "sparse", N, t_sysS_diag / denom);
        write_row(out, "symm",        "sparse", N, t_sysS_symm / denom);
        write_row(out, "matvec",      "sparse", N, t_sysS_matvec / denom);
        write_row(out, "residual",    "sparse", N, t_sysS_res / denom);
    }

    std::cout << "Benchmark complete → bench_random.csv\n";
    if (sink == 123456789.0) std::cerr << sink << "\n";
    return 0;
}