#include <iostream>
#include <cmath>

#include "DenseSquareMatrixDouble.hpp"
#include "SparseSquareMatrixCRSDouble.hpp"
#include "VectorDouble.hpp"
#include "matIO.hpp"

int main()
{
    const std::size_t N = 5;

    // ===============================
    // Dense test + write for plotting
    // ===============================
    DenseSquareMatrixDouble A(N);
    for (std::size_t i = 0; i < N; ++i) {
        A(i,i) = 2.0;
        if (i>0)   A(i,i-1) = -1.0;
        if (i<N-1) A(i,i+1) = -1.0;
    }

    // Write matrix for visualization
    matIO::writeMTX("dense_A.mtx", A);

    DenseSquareMatrixDouble A2(N);
    matIO::readMTX("dense_A.mtx", A2);

    std::cout << "Dense equal? "
              << (A == A2 ? "YES" : "NO")
              << "\n";

    // ===============================
    // Sparse test + write for plotting
    // ===============================
    SparseSquareMatrixCRSDouble S(N);
    for (std::size_t i = 0; i < N; ++i) {
        S.addEntry(i,i,2.0);
        if (i>0)   S.addEntry(i,i-1,-1.0);
        if (i<N-1) S.addEntry(i,i+1,-1.0);
    }
    S.finalize();

    // Write sparse matrix for visualization
    matIO::writeMTX("sparse_A.mtx", S);

    SparseSquareMatrixCRSDouble S2(N);
    matIO::readMTX("sparse_A.mtx", S2);

    std::cout << "Sparse equal? "
              << (S == S2 ? "YES" : "NO")
              << "\n";

    // ===============================
    // Vector test + write for plotting
    // ===============================
    VectorDouble x_true(N);
    for (std::size_t i = 0; i < N; ++i)
        x_true[i] = static_cast<double>(i + 1);  // 1,2,3,4,5

    // compute b = A * x_true  (dense multiply, in main for now)
    VectorDouble b(N);
    for (std::size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < N; ++j)
            sum += A(i,j) * x_true[j];
        b[i] = sum;
    }

    // compute residual r = b - A*x_true (should be 0 vector)
    VectorDouble Ax(N);
    for (std::size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < N; ++j)
            sum += A(i,j) * x_true[j];
        Ax[i] = sum;
    }

    VectorDouble r = b - Ax;

    // Write vectors for visualization
    matIO::writeMTX("x_true.mtx", x_true);
    matIO::writeMTX("b.mtx", b);
    matIO::writeMTX("residual.mtx", r);

    VectorDouble b2(N);
    matIO::readMTX("b.mtx", b2);
    std::cout << "Vector b equal after IO? "
              << (b == b2 ? "YES" : "NO")
              << "\n";

    std::cout << "Wrote files:\n"
              << "  dense_A.mtx\n"
              << "  sparse_A.mtx\n"
              << "  x_true.mtx\n"
              << "  b.mtx\n"
              << "  residual.mtx\n";

    return 0;
}