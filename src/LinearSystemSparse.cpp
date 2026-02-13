#include "LinearSystemSparse.hpp"
#include <stdexcept>
#include <utility>

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
