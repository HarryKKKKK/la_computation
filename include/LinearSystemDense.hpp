#pragma once
#include "DenseSquareMatrixDouble.hpp"
#include "VectorDouble.hpp"

class LinearSystemDense {
public:
    explicit LinearSystemDense(DenseSquareMatrixDouble&& A, VectorDouble&& x, VectorDouble&& b);

    DenseSquareMatrixDouble& A();
    VectorDouble& x();
    VectorDouble& b();

    const DenseSquareMatrixDouble& A() const;
    const VectorDouble& x() const;
    const VectorDouble& b() const;

    // compute b = A * x
    void multiply();
    // r = b - A * x
    VectorDouble residual() const;
    // solve x = A / b
    // VectorDouble solve() const;

    bool isSymmetric() const;
    bool isDiagonallyDominant() const;

private:
    DenseSquareMatrixDouble A_;
    VectorDouble x_;
    VectorDouble b_;
};
