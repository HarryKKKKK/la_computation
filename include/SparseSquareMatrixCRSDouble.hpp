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
    bool isFinalized() const noexcept { return finalized_; }

    const std::vector<std::size_t>& rowPtr() const { return rowPtr_; }
    const std::vector<std::size_t>& colInd() const { return colInd_; }
    const std::vector<double>& values() const { return val_; }
    const VectorDouble& diagonal() const { return diag_; }

    SparseSquareMatrixCRSDouble operator+(const SparseSquareMatrixCRSDouble& other) const;
    SparseSquareMatrixCRSDouble operator-(const SparseSquareMatrixCRSDouble& other) const;
    SparseSquareMatrixCRSDouble operator*(double scalar) const;
    VectorDouble operator*(const VectorDouble& x) const;

    bool operator==(const SparseSquareMatrixCRSDouble& other) const;
    bool operator!=(const SparseSquareMatrixCRSDouble& other) const;

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

    // CRS storage
    std::vector<std::size_t> rowPtr_;
    std::vector<std::size_t> colInd_;
    std::vector<double> val_;
    VectorDouble diag_;
};
