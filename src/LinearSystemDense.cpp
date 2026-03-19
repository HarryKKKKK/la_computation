#include "LinearSystemDense.hpp"
#include <stdexcept>
#include <omp.h>

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

VectorDouble LinearSystemDense::residual_opt() const
{
    const std::size_t N = A_.size();
    VectorDouble r(N);

    const double* const A_ptr = &A_(0, 0); 
    const double* const x_ptr = &x_[0];
    const double* const b_ptr = &b_[0];
    double* const r_ptr = &r[0];

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
    {
        double Ax_i = 0.0;
        
        const double* const row_ptr = &A_ptr[i * N];
        for (std::size_t j = 0; j < N; ++j)
        {
            Ax_i += row_ptr[j] * x_ptr[j];
        }

        r_ptr[i] = b_ptr[i] - Ax_i;
    }

    return r;
}

bool LinearSystemDense::isSymmetric() const
{
    return A_.isSymmetric();
}

bool LinearSystemDense::isDiagonallyDominant() const
{
    return A_.isDiagonallyDominant();
}

// ===============================
// Arith Op
// ===============================
LinearSystemDense
LinearSystemDense::operator+(const LinearSystemDense& other) const
{
    if (A_.size() != other.A_.size())
        throw std::runtime_error("Dimension mismatch in LinearSystemDense +");

    DenseSquareMatrixDouble Anew = A_ + other.A_;
    VectorDouble bnew = b_ + other.b_;

    VectorDouble xnew = x_;

    return LinearSystemDense(std::move(Anew),
                             std::move(xnew),
                             std::move(bnew));
}

LinearSystemDense
LinearSystemDense::operator-(const LinearSystemDense& other) const
{
    if (A_.size() != other.A_.size())
        throw std::runtime_error("Dimension mismatch in LinearSystemDense -");

    DenseSquareMatrixDouble Anew = A_ - other.A_;
    VectorDouble bnew = b_ - other.b_;

    VectorDouble xnew = x_;

    return LinearSystemDense(std::move(Anew),
                             std::move(xnew),
                             std::move(bnew));
}

LinearSystemDense
LinearSystemDense::operator*(double scalar) const
{
    DenseSquareMatrixDouble Anew = A_ * scalar;
    VectorDouble bnew = b_ * scalar;

    VectorDouble xnew = x_;

    return LinearSystemDense(std::move(Anew),
                             std::move(xnew),
                             std::move(bnew));
}