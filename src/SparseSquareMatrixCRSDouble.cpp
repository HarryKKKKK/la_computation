#include "SparseSquareMatrixCRSDouble.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

SparseSquareMatrixCRSDouble::SparseSquareMatrixCRSDouble(std::size_t N)
    : N_(N), finalized_(false), diag_(N)
{}

std::size_t SparseSquareMatrixCRSDouble::size() const noexcept { return N_; }
std::size_t SparseSquareMatrixCRSDouble::nnz()  const noexcept { return val_.size(); }

void SparseSquareMatrixCRSDouble::addEntry(std::size_t i, std::size_t j, double val)
{
    if (finalized_)
        throw std::runtime_error("Error: Cannot addEntry after finalize()");
    if (i >= N_ || j >= N_)
        throw std::runtime_error("Error: addEntry index out of range");

    entries_.push_back({i, j, val});
}

void SparseSquareMatrixCRSDouble::finalize()
{
    if (finalized_)
        return;

    // Reset CRS storage
    for (std::size_t i = 0; i < N_; ++i) {
        diag_[i] = 0.0;
    }
    rowPtr_.assign(N_ + 1, 0);
    colInd_.clear();
    val_.clear();

    // Sort triplets by (row, col)
    std::sort(entries_.begin(), entries_.end(),
              [](const Triplet& a, const Triplet& b) {
                  if (a.i != b.i) {
                    return a.i < b.i;
                  }
                  return a.j < b.j;
              });

    // First pass: count unique OFF-diagonal entries per row,
    std::size_t k = 0;
    while (k < entries_.size()) {
        std::size_t i = entries_[k].i;
        std::size_t j = entries_[k].j;
        double sum = entries_[k].v;

        std::size_t k2 = k + 1;
        while (k2 < entries_.size() && entries_[k2].i == i && entries_[k2].j == j) {
            sum += entries_[k2].v;
            ++k2;
        }

        if (i == j) {
            diag_[i] += sum;
        } else {
            rowPtr_[i + 1] += 1; // one unique off-diag entry in row i
        }

        k = k2;
    }

    // Prefix sum to build rowPtr
    for (std::size_t i = 0; i < N_; ++i)
        rowPtr_[i + 1] += rowPtr_[i];

    const std::size_t nnz_off = rowPtr_[N_];
    colInd_.assign(nnz_off, 0);
    val_.assign(nnz_off, 0.0);

    // Second pass: fill colInd/val for OFF-diagonal
    std::vector<std::size_t> cursor = rowPtr_;

    k = 0;
    while (k < entries_.size()) {
        std::size_t i = entries_[k].i;
        std::size_t j = entries_[k].j;
        double sum = entries_[k].v;

        std::size_t k2 = k + 1;
        while (k2 < entries_.size() && entries_[k2].i == i && entries_[k2].j == j) {
            sum += entries_[k2].v;
            ++k2;
        }

        if (i != j) {
            std::size_t pos = cursor[i]++;
            colInd_[pos] = j;
            val_[pos] = sum;
        }

        k = k2;
    }

    finalized_ = true;

    entries_.clear();
    entries_.shrink_to_fit();
}

VectorDouble SparseSquareMatrixCRSDouble::operator*(const VectorDouble& x) const
{
    if (!finalized_)
        throw std::runtime_error("Error: SparseSquareMatrixCRSDouble not finalized()");
    if (x.size() != N_)
        throw std::runtime_error("Error: Dimension mismatch in sparse A*x");

    VectorDouble y(N_);

    for (std::size_t i = 0; i < N_; ++i) {
        double sum = diag_[i] * x[i];

        for (std::size_t p = rowPtr_[i]; p < rowPtr_[i + 1]; ++p) {
            const std::size_t j = colInd_[p];
            sum += val_[p] * x[j];
        }

        y[i] = sum;
    }

    return y;
}
