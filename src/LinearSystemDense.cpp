#include "LinearSystemDense.hpp"
#include <stdexcept>

#include <cmath>
#include <iostream>

LinearSystemDense::LinearSystemDense(DenseSquareMatrixDouble&& A,
                                     VectorDouble&& x,
                                     VectorDouble&& b)
    : A_(std::move(A)), x_(std::move(x)), b_(std::move(b))
{
    if (A_.size() != x_.size() || A_.size() != b_.size())
        throw std::runtime_error("Error: Dimension mismatch in LinearSystemDense constructor");
}

DenseSquareMatrixDouble& LinearSystemDense::A() { return A_; }
VectorDouble& LinearSystemDense::x() { return x_; }
VectorDouble& LinearSystemDense::b() { return b_; }

const DenseSquareMatrixDouble& LinearSystemDense::A() const { return A_; }
const VectorDouble& LinearSystemDense::x() const { return x_; }
const VectorDouble& LinearSystemDense::b() const { return b_; }

void LinearSystemDense::multiply()
{
    b_ = A_ * x_;
}

VectorDouble LinearSystemDense::residual() const
{
    return b_ - (A_ * x_);
}

bool LinearSystemDense::isSymmetric() const
{
    // std::cout << "check for symmetry" << std::endl;
    double TOL = 1e-12;
    const std::size_t N = A_.size();

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            // std::cout << i << " " << j << " " << std::abs(A_(i, j) - A_(j, i)) << std::endl;
            if (std::abs(A_(i, j) - A_(j, i)) > TOL) {
                // std::cout << std::abs(A_(i, j) - A_(j, i)) << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool LinearSystemDense::isDiagonallyDominant() const
{
    const std::size_t N = A_.size();

    for (std::size_t i = 0; i < N; ++i) {
        double diag = std::abs(A_(i, i));
        double off_sum = 0.0;

        for (std::size_t j = 0; j < N; ++j) {
            if (j == i) continue;
            off_sum += std::abs(A_(i, j));
        }

        if (diag < off_sum)
            return false;
    }
    return true;
}
