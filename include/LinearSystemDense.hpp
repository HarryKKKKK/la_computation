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

    LinearSystemDense operator+(const LinearSystemDense& other) const;
    LinearSystemDense operator-(const LinearSystemDense& other) const;
    LinearSystemDense operator*(double scalar) const;

    // b = A * x
    void multiply();
    // r = b - A * x
    VectorDouble residual() const;
    VectorDouble residual_opt() const;

    bool isSymmetric() const;
    bool isDiagonallyDominant() const;

private:
    DenseSquareMatrixDouble A_;
    VectorDouble x_;
    VectorDouble b_;
};
