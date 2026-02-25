#include <iostream>
#include <cmath>
#include <stdexcept>

#include "SparseSquareMatrixCRSDouble.hpp"
#include "DenseSquareMatrixDouble.hpp"
#include "VectorDouble.hpp"

static void expect_near(double a, double b, double tol, const char* msg)
{
    if (std::abs(a - b) > tol) {
        std::cerr << "[FAIL] " << msg << " | got " << a
                  << " expected " << b << "\n";
        std::exit(1);
    }
}

static void expect_true(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

static void test_identity()
{
    std::cout << "Running sparse identity test...\n";

    const std::size_t N = 4;
    SparseSquareMatrixCRSDouble A(N);

    for (std::size_t i = 0; i < N; ++i)
        A.addEntry(i, i, 1.0);

    A.finalize();

    VectorDouble x(N);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = double(i + 1);

    VectorDouble b = A * x;

    for (std::size_t i = 0; i < N; ++i)
        expect_near(b[i], x[i], 1e-12, "Identity A*x = x");

    std::cout << "  OK\n";
}

static void test_diagonal()
{
    std::cout << "Running sparse diagonal test...\n";

    SparseSquareMatrixCRSDouble A(3);

    A.addEntry(0,0,2.0);
    A.addEntry(1,1,3.0);
    A.addEntry(2,2,4.0);

    A.finalize();

    VectorDouble x(3);
    x[0]=1; x[1]=2; x[2]=3;

    VectorDouble b = A * x;

    expect_near(b[0], 2.0, 1e-12, "diag[0]");
    expect_near(b[1], 6.0, 1e-12, "diag[1]");
    expect_near(b[2], 12.0, 1e-12, "diag[2]");

    std::cout << "  OK\n";
}

static void test_tridiagonal()
{
    std::cout << "Running sparse tridiagonal test...\n";

    const std::size_t N = 5;
    SparseSquareMatrixCRSDouble A(N);

    for (std::size_t i = 0; i < N; ++i) {
        A.addEntry(i, i, 2.0);
        if (i > 0)     A.addEntry(i, i-1, -1.0);
        if (i < N-1)   A.addEntry(i, i+1, -1.0);
    }

    A.finalize();

    VectorDouble x(N);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = 1.0;

    VectorDouble b = A * x;

    // For interior rows: 2*1 -1*1 -1*1 = 0
    // For boundaries: 2*1 -1*1 = 1
    expect_near(b[0], 1.0, 1e-12, "tri[0]");
    expect_near(b[1], 0.0, 1e-12, "tri[1]");
    expect_near(b[2], 0.0, 1e-12, "tri[2]");
    expect_near(b[3], 0.0, 1e-12, "tri[3]");
    expect_near(b[4], 1.0, 1e-12, "tri[4]");

    std::cout << "  OK\n";
}

static void test_dense_vs_sparse()
{
    std::cout << "Running dense vs sparse comparison...\n";

    const std::size_t N = 5;

    DenseSquareMatrixDouble Ad(N);
    SparseSquareMatrixCRSDouble As(N);

    // build same tridiagonal
    for (std::size_t i = 0; i < N; ++i) {
        Ad(i,i) = 2.0;
        As.addEntry(i,i,2.0);

        if (i > 0) {
            Ad(i,i-1) = -1.0;
            As.addEntry(i,i-1,-1.0);
        }
        if (i < N-1) {
            Ad(i,i+1) = -1.0;
            As.addEntry(i,i+1,-1.0);
        }
    }

    As.finalize();

    VectorDouble x(N);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = double(i+1);

    VectorDouble bd = Ad * x;
    VectorDouble bs = As * x;

    VectorDouble diff = bd - bs;

    expect_true(diff.normInf() < 1e-12, "Dense vs Sparse mismatch");

    std::cout << "  OK\n";
}

int main()
{
    test_identity();
    test_diagonal();
    test_tridiagonal();
    test_dense_vs_sparse();

    std::cout << "\nAll sparse tests PASSED\n";
    return 0;
}