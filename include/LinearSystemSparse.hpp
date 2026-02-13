#pragma once
#include <cstddef>
#include "SparseSquareMatrixCRSDouble.hpp"
#include "VectorDouble.hpp"

// Linear equation set: A x = b, where A is sparse CRS (double)
class LinearSystemSparse {
public:
    explicit LinearSystemSparse(SparseSquareMatrixCRSDouble&& A,
                                VectorDouble&& x,
                                VectorDouble&& b);

    SparseSquareMatrixCRSDouble& A();
    VectorDouble& x();
    VectorDouble& b();

    const SparseSquareMatrixCRSDouble& A() const;
    const VectorDouble& x() const;
    const VectorDouble& b() const;

    void multiply();
    VectorDouble residual() const;

private:
    SparseSquareMatrixCRSDouble A_;
    VectorDouble x_;
    VectorDouble b_;
};
