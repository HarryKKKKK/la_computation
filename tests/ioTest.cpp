#include <iostream>
#include <cmath>
#include "DenseSquareMatrixDouble.hpp"
#include "SparseSquareMatrixCRSDouble.hpp"
#include "matIO.hpp"

int main()
{
    const std::size_t N = 5;

    // ===============================
    // Dense test
    // ===============================
    DenseSquareMatrixDouble A(N);
    for (std::size_t i = 0; i < N; ++i) {
        A(i,i) = 2.0;
        if (i>0)   A(i,i-1) = -1.0;
        if (i<N-1) A(i,i+1) = -1.0;
    }

    matIO::writeMTX("dense.mtx", A);

    DenseSquareMatrixDouble A2(N);
    matIO::readMTX("dense.mtx", A2);

    std::cout << "Dense equal? "
              << (A == A2 ? "YES" : "NO")
              << "\n";

    // ===============================
    // Sparse test
    // ===============================
    SparseSquareMatrixCRSDouble S(N);
    for (std::size_t i = 0; i < N; ++i) {
        S.addEntry(i,i,2.0);
        if (i>0)   S.addEntry(i,i-1,-1.0);
        if (i<N-1) S.addEntry(i,i+1,-1.0);
    }
    S.finalize();

    matIO::writeMTX("sparse.mtx", S);

    SparseSquareMatrixCRSDouble S2(N);
    matIO::readMTX("sparse.mtx", S2);

    std::cout << "Sparse equal? "
              << (S == S2 ? "YES" : "NO")
              << "\n";

    return 0;
}