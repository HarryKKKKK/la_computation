#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <utility>
#include "VectorDouble.hpp"

class SparseSquareMatrixCRSDouble {
public:
    explicit SparseSquareMatrixCRSDouble(std::size_t N);

    std::size_t size() const noexcept;
    std::size_t nnz() const noexcept; 

    void addEntry(std::size_t i, std::size_t j, double val);
    void finalize();

    VectorDouble operator*(const VectorDouble& x) const;

    const std::vector<std::size_t>& rowPtr() const { return rowPtr_; }
    const std::vector<std::size_t>& colInd() const { return colInd_; }
    const std::vector<double>& values() const { return val_; }
    const VectorDouble& diagonal() const { return diag_; }

private:
    struct Triplet {
        std::size_t i;
        std::size_t j;
        double v;
    };

    std::size_t N_;

    // builder storage
    std::vector<Triplet> entries_;
    bool finalized_;

    // CRS storage for OFF-diagonal entries only
    std::vector<std::size_t> rowPtr_;  // size N_+1
    std::vector<std::size_t> colInd_;  // size nnz
    std::vector<double> val_;     // size nnz

    // diagonal stored separately
    VectorDouble diag_;
};
