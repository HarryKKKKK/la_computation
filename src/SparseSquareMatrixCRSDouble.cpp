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

bool SparseSquareMatrixCRSDouble::operator==(const SparseSquareMatrixCRSDouble& other) const
{
    if (N_ != other.N_) return false;
    if (!finalized_ || !other.finalized_) return false;

    const double tol = 1e-12;

    // 1) diagonal
    for (std::size_t i = 0; i < N_; ++i) {
        if (std::abs(diag_[i] - other.diag_[i]) > tol)
            return false;
    }

    // 2) rowPtr exact (should match if both are canonical CSR)
    if (rowPtr_.size() != other.rowPtr_.size()) return false;
    for (std::size_t i = 0; i < rowPtr_.size(); ++i) {
        if (rowPtr_[i] != other.rowPtr_[i])
            return false;
    }

    // 3) colInd exact
    if (colInd_.size() != other.colInd_.size()) return false;
    for (std::size_t k = 0; k < colInd_.size(); ++k) {
        if (colInd_[k] != other.colInd_[k])
            return false;
    }

    // 4) values with tolerance
    if (val_.size() != other.val_.size()) return false;
    for (std::size_t k = 0; k < val_.size(); ++k) {
        if (std::abs(val_[k] - other.val_[k]) > tol)
            return false;
    }

    return true;
}

bool SparseSquareMatrixCRSDouble::operator!=(const SparseSquareMatrixCRSDouble& other) const
{
    return !(*this == other);
}

SparseSquareMatrixCRSDouble
SparseSquareMatrixCRSDouble::operator+(
    const SparseSquareMatrixCRSDouble& other) const
{
    if (N_ != other.N_)
        throw std::runtime_error("Sparse + dimension mismatch");

    if (!finalized_ || !other.finalized_)
        throw std::runtime_error("Matrices must be finalized");

    SparseSquareMatrixCRSDouble C(N_);

    for (std::size_t i = 0; i < N_; ++i)
    {
        // diagonal
        C.addEntry(i, i, diag_[i] + other.diag_[i]);

        // this matrix
        for (std::size_t p = rowPtr_[i]; p < rowPtr_[i+1]; ++p)
            C.addEntry(i, colInd_[p], val_[p]);

        // other matrix
        for (std::size_t p = other.rowPtr_[i]; p < other.rowPtr_[i+1]; ++p)
            C.addEntry(i, other.colInd_[p], other.val_[p]);
    }

    C.finalize();
    return C;
}

SparseSquareMatrixCRSDouble
SparseSquareMatrixCRSDouble::operator-(
    const SparseSquareMatrixCRSDouble& other) const
{
    if (N_ != other.N_)
        throw std::runtime_error("Sparse - dimension mismatch");

    if (!finalized_ || !other.finalized_)
        throw std::runtime_error("Matrices must be finalized");

    SparseSquareMatrixCRSDouble C(N_);

    for (std::size_t i = 0; i < N_; ++i)
    {
        C.addEntry(i, i, diag_[i] - other.diag_[i]);

        for (std::size_t p = rowPtr_[i]; p < rowPtr_[i+1]; ++p)
            C.addEntry(i, colInd_[p], val_[p]);

        for (std::size_t p = other.rowPtr_[i]; p < other.rowPtr_[i+1]; ++p)
            C.addEntry(i, other.colInd_[p], -other.val_[p]);
    }

    C.finalize();
    return C;
}

SparseSquareMatrixCRSDouble
SparseSquareMatrixCRSDouble::operator*(double scalar) const
{
    if (!finalized_)
        throw std::runtime_error("Matrix must be finalized");

    SparseSquareMatrixCRSDouble C(N_);

    for (std::size_t i = 0; i < N_; ++i)
    {
        C.addEntry(i, i, diag_[i] * scalar);

        for (std::size_t p = rowPtr_[i]; p < rowPtr_[i+1]; ++p)
            C.addEntry(i, colInd_[p], val_[p] * scalar);
    }

    C.finalize();
    return C;
}
bool SparseSquareMatrixCRSDouble::isDiagonallyDominant() const
{
    if (!finalized_)
        throw std::runtime_error("Error: Matrix must be finalized before checks");

    const double tol = 1e-12;

    for (std::size_t i = 0; i < N_; ++i)
    {
        const double diagAbs = std::abs(diag_[i]);
        double offSum = 0.0;

        for (std::size_t p = rowPtr_[i]; p < rowPtr_[i + 1]; ++p)
            offSum += std::abs(val_[p]);

        if (diagAbs + tol < offSum)
            return false;
    }

    return true;
}

bool SparseSquareMatrixCRSDouble::isSymmetric() const
{
    if (!finalized_)
        throw std::runtime_error("Error: Matrix must be finalized before checks");

    const double tol = 1e-12;

    const std::size_t nnz_off = val_.size();

    // Build transpose CRS of off-diagonal part WITHOUT sorting:
    // Because we traverse i from 0..N-1 and inside each row colInd_ is increasing,
    // the inserted "column indices" (which are i) into each transpose row will be increasing.
    std::vector<std::size_t> tRowPtr(N_ + 1, 0);

    // 1) count entries per transpose row (row = original column j)
    for (std::size_t p = 0; p < nnz_off; ++p) {
        const std::size_t j = colInd_[p];
        tRowPtr[j + 1] += 1;
    }

    // 2) prefix sum
    for (std::size_t r = 0; r < N_; ++r)
        tRowPtr[r + 1] += tRowPtr[r];

    std::vector<std::size_t> tColInd(nnz_off, 0);
    std::vector<double>      tVal(nnz_off, 0.0);

    // 3) stable fill using row-major traversal of A
    std::vector<std::size_t> cursor = tRowPtr;
    for (std::size_t i = 0; i < N_; ++i) {
        for (std::size_t p = rowPtr_[i]; p < rowPtr_[i + 1]; ++p) {
            const std::size_t j = colInd_[p];
            const double aij = val_[p];

            const std::size_t pos = cursor[j]++;
            tColInd[pos] = i;   // transpose has column = original row i
            tVal[pos]    = aij;
        }
    }

    // 4) merge-compare each row i: A(i,*) vs A^T(i,*)
    for (std::size_t i = 0; i < N_; ++i) {
        std::size_t p  = rowPtr_[i];
        std::size_t pe = rowPtr_[i + 1];
        std::size_t q  = tRowPtr[i];
        std::size_t qe = tRowPtr[i + 1];

        while (p < pe && q < qe) {
            const std::size_t cA  = colInd_[p];
            const std::size_t cAt = tColInd[q];

            if (cA == cAt) {
                if (std::abs(val_[p] - tVal[q]) > tol) return false;
                ++p; ++q;
            } else if (cA < cAt) {
                // A has an entry that transpose-row lacks => asymmetric unless it's ~0
                if (std::abs(val_[p]) > tol) return false;
                ++p;
            } else { // cAt < cA
                if (std::abs(tVal[q]) > tol) return false;
                ++q;
            }
        }

        while (p < pe) { if (std::abs(val_[p]) > tol) return false; ++p; }
        while (q < qe) { if (std::abs(tVal[q]) > tol) return false; ++q; }
    }

    // diagonal is trivially symmetric
    return true;
}