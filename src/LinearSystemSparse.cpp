#include "LinearSystemSparse.hpp"
#include <stdexcept>
#include <utility>

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
    const std::size_t N = A_.size();

    const auto& rp = A_.rowPtr();
    const auto& va = A_.values();
    const auto& diagnal = A_.diagonal();

    for (std::size_t i = 0; i < N; ++i)
    {
        double diag = std::abs(diagnal[i]);
        double off_sum = 0.0;

        for (std::size_t p = rp[i]; p < rp[i + 1]; ++p)
            off_sum += std::abs(va[p]);

        if (diag < off_sum)
            return false;
    }
    return true;
}

bool LinearSystemSparse::isSymmetric() const
{
    if (!A_.isFinalized()) {
        throw std::runtime_error("Error: Matrix must be finalized");
    }
    const double TOL = 1e-12;
    const std::size_t N = A_.size();

    const auto& rp = A_.rowPtr();
    const auto& ci = A_.colInd();
    const auto& va = A_.values();

    // For each (i,j) in off-diagonal, find (j,i)
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t p = rp[i]; p < rp[i + 1]; ++p)
        {
            const std::size_t j = ci[p];
            const double aij = va[p];

            // look for aji in row j
            bool found = false;
            double aji = 0.0;

            for (std::size_t q = rp[j]; q < rp[j + 1]; ++q)
            {
                if (ci[q] == i)
                {
                    found = true;
                    aji = va[q];
                    break;
                }
            }

            if (!found)
                return false;

            if (std::abs(aij - aji) > TOL)
                return false;
        }
    }

    return true;
}
