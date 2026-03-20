#include "SparseSquareMatrixCRSDouble.hpp"
#include "DenseSquareMatrixDouble.hpp"
#include "matIO.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

SparseSquareMatrixCRSDouble buildComplexSparseMatrix(std::size_t N)
{
    SparseSquareMatrixCRSDouble A(N);

    // --------------------------------------------------
    // 1. Add off-diagonal couplings first
    // --------------------------------------------------

    // (a) First off-diagonal band: i <-> i+1
    for (std::size_t i = 0; i + 1 < N; ++i) {
        A.addEntry(i, i + 1, -1.0);
        A.addEntry(i + 1, i, -1.0);
    }

    // (b) Second off-diagonal band: i <-> i+2
    for (std::size_t i = 0; i + 2 < N; ++i) {
        A.addEntry(i, i + 2, -0.35);
        A.addEntry(i + 2, i, -0.35);
    }

    // (c) Block-local dense-ish structure
    const std::size_t blockSize = 12;
    for (std::size_t b = 0; b < N; b += blockSize) {
        const std::size_t end = std::min(N, b + blockSize);

        for (std::size_t i = b; i < end; ++i) {
            for (std::size_t j = i + 3; j < end; j += 3) {
                const double w = -0.18;
                A.addEntry(i, j, w);
                A.addEntry(j, i, w);
            }
        }
    }

    // (d) Add deterministic long-range couplings for visual complexity
    for (std::size_t i = 0; i < N; i += 11) {
        std::size_t j = (3 * i + 17) % N;
        if (i != j) {
            const double w = -0.12;
            A.addEntry(i, j, w);
            A.addEntry(j, i, w);
        }
    }

    std::vector<double> offAbsSum(N, 0.0);

    // Recreate the same pattern and accumulate magnitudes row by row.

    // First band
    for (std::size_t i = 0; i + 1 < N; ++i) {
        offAbsSum[i]     += 1.0;
        offAbsSum[i + 1] += 1.0;
    }

    // Second band
    for (std::size_t i = 0; i + 2 < N; ++i) {
        offAbsSum[i]     += 0.35;
        offAbsSum[i + 2] += 0.35;
    }

    // Block-local structure
    for (std::size_t b = 0; b < N; b += blockSize) {
        const std::size_t end = std::min(N, b + blockSize);

        for (std::size_t i = b; i < end; ++i) {
            for (std::size_t j = i + 3; j < end; j += 3) {
                offAbsSum[i] += 0.18;
                offAbsSum[j] += 0.18;
            }
        }
    }

    // Long-range couplings
    for (std::size_t i = 0; i < N; i += 11) {
        std::size_t j = (3 * i + 17) % N;
        if (i != j) {
            offAbsSum[i] += 0.12;
            offAbsSum[j] += 0.12;
        }
    }

    // Add diagonal with a small positive safety margin
    for (std::size_t i = 0; i < N; ++i) {
        A.addEntry(i, i, offAbsSum[i] + 1.0);
    }

    A.finalize();
    return A;
}


int main()
{
    try {
        const std::size_t N = 120;

        // Build sparse matrix
        SparseSquareMatrixCRSDouble A = buildComplexSparseMatrix(N);

        // Write sparse matrix for sparsity-pattern plotting
        matIO::writeMTX("complex_sparse_matrix.mtx", A);
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}