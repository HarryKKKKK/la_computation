#pragma once
#include <cstddef>
#include <memory>
#include "VectorDouble.hpp"

// FIXME: Dense square matrix of size n x n of Doubles, could be exteneded
class DenseSquareMatrixDouble {
public:
    explicit DenseSquareMatrixDouble(std::size_t N);

    // basic operation
    DenseSquareMatrixDouble(const DenseSquareMatrixDouble& other);
    DenseSquareMatrixDouble& operator=(const DenseSquareMatrixDouble& other);
    DenseSquareMatrixDouble(DenseSquareMatrixDouble&& other) noexcept;
    DenseSquareMatrixDouble& operator=(DenseSquareMatrixDouble&& other) noexcept;
    ~DenseSquareMatrixDouble() = default;

    std::size_t size() const noexcept;

    // element access: A(i, j)
    double& operator()(std::size_t i, std::size_t j);
    const double& operator()(std::size_t i, std::size_t j) const;

    // algebra
    DenseSquareMatrixDouble operator+(const DenseSquareMatrixDouble& other) const;
    DenseSquareMatrixDouble operator-(const DenseSquareMatrixDouble& other) const;
    DenseSquareMatrixDouble operator*(const DenseSquareMatrixDouble& other) const;
    DenseSquareMatrixDouble operator*(double scalar) const;
    VectorDouble operator*(const VectorDouble& x) const;

private:
    std::size_t N_;
    // A[i, j]: data_[i * N_ + j]
    std::unique_ptr<double[]> data_;
};
