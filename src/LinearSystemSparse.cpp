#include "LinearSystemSparse.hpp"
#include <stdexcept>
#include <utility>
#include <omp.h>

#include <cmath>

LinearSystemSparse::LinearSystemSparse(SparseSquareMatrixCRSDouble&& A,
                                       VectorDouble&& x,
                                       VectorDouble&& b)
    : A_(std::move(A)), x_(std::move(x)), b_(std::move(b))
{
    const std::size_t N = A_.size();
    if (x_.size() != N || b_.size() != N)
        throw std::runtime_error("Dimension mismatch in LinearSystemSparse constructor");
}

SparseSquareMatrixCRSDouble& LinearSystemSparse::A() { return A_; }
VectorDouble& LinearSystemSparse::x() { return x_; }
VectorDouble& LinearSystemSparse::b() { return b_; }

const SparseSquareMatrixCRSDouble& LinearSystemSparse::A() const { return A_; }
const VectorDouble& LinearSystemSparse::x() const { return x_; }
const VectorDouble& LinearSystemSparse::b() const { return b_; }

void LinearSystemSparse::multiply()
{
    b_ = A_ * x_;
}

VectorDouble LinearSystemSparse::residual() const
{
    return b_ - (A_ * x_);
}

VectorDouble LinearSystemSparse::residual_opt() const {
    const std::size_t N = A_.size();
    VectorDouble r(N); 

    const auto& rowPtr = A_.rowPtr();
    const auto& colInd = A_.colInd();
    const auto& val = A_.values();
    const auto& diag = A_.diagonal();

    const double* const x_ptr = &x_[0];
    const double* const b_ptr = &b_[0];
    double* const r_ptr = &r[0];

    for (std::size_t i = 0; i < N; ++i) {
        double Ax_i = diag[i] * x_ptr[i];

        const std::size_t p_start = rowPtr[i];
        const std::size_t p_end = rowPtr[i + 1];

        for (std::size_t p = p_start; p < p_end; ++p) {
            Ax_i += val[p] * x_ptr[colInd[p]];
        }
        
        r_ptr[i] = b_ptr[i] - Ax_i;
    }

    return r;
}

// ===============================
// Arthematic Op
// ===============================
LinearSystemSparse
LinearSystemSparse::operator+(const LinearSystemSparse& other) const
{
    if (A_.size() != other.A_.size())
        throw std::runtime_error("Dimension mismatch in LinearSystemSparse +");

    SparseSquareMatrixCRSDouble Anew = A_ + other.A_;
    VectorDouble bnew = b_ + other.b_;
    VectorDouble xnew = x_;

    return LinearSystemSparse(std::move(Anew),
                              std::move(xnew),
                              std::move(bnew));
}

LinearSystemSparse
LinearSystemSparse::operator-(const LinearSystemSparse& other) const
{
    if (A_.size() != other.A_.size())
        throw std::runtime_error("Dimension mismatch in LinearSystemSparse -");

    SparseSquareMatrixCRSDouble Anew = A_ - other.A_;
    VectorDouble bnew = b_ - other.b_;

    VectorDouble xnew = x_;

    return LinearSystemSparse(std::move(Anew),
                              std::move(xnew),
                              std::move(bnew));
}

LinearSystemSparse
LinearSystemSparse::operator*(double scalar) const
{
    SparseSquareMatrixCRSDouble Anew = A_ * scalar;
    VectorDouble bnew = b_ * scalar;

    VectorDouble xnew = x_;

    return LinearSystemSparse(std::move(Anew),
                              std::move(xnew),
                              std::move(bnew));
}


bool LinearSystemSparse::isDiagonallyDominant() const
{
    return A_.isDiagonallyDominant();
}

bool LinearSystemSparse::isSymmetric() const
{
    return A_.isSymmetric();
}
