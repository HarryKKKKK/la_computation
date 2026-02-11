#include "DenseSquareMatrixDouble.hpp"
#include <stdexcept>
#include <utility>

DenseSquareMatrixDouble::DenseSquareMatrixDouble(std::size_t N)
    : N_(N), data_(std::make_unique<double[]>(N * N))
{
    for (std::size_t i = 0; i < N_ * N_; ++i)
        data_[i] = 0.0;
}

DenseSquareMatrixDouble::DenseSquareMatrixDouble(const DenseSquareMatrixDouble& other)
    : N_(other.N_), data_(std::make_unique<double[]>(other.N_ * other.N_))
{
    for (std::size_t i = 0; i < N_ * N_; ++i)
        data_[i] = other.data_[i];
}

DenseSquareMatrixDouble&
DenseSquareMatrixDouble::operator=(const DenseSquareMatrixDouble& other)
{
    if (this == &other)
        return *this;

    if (N_ != other.N_) {
        N_ = other.N_;
        data_ = std::make_unique<double[]>(N_ * N_);
    }

    for (std::size_t i = 0; i < N_ * N_; ++i)
        data_[i] = other.data_[i];

    return *this;
}

DenseSquareMatrixDouble::DenseSquareMatrixDouble(DenseSquareMatrixDouble&& other) noexcept
    : N_(other.N_), data_(std::move(other.data_))
{
    other.N_ = 0;
}

DenseSquareMatrixDouble&
DenseSquareMatrixDouble::operator=(DenseSquareMatrixDouble&& other) noexcept
{
    if (this == &other)
        return *this;

    N_ = other.N_;
    data_ = std::move(other.data_);
    other.N_ = 0;

    return *this;
}

std::size_t DenseSquareMatrixDouble::size() const noexcept
{
    return N_;
}

double& DenseSquareMatrixDouble::operator()(std::size_t i, std::size_t j)
{
    return data_[i * N_ + j];
}

const double& DenseSquareMatrixDouble::operator()(std::size_t i, std::size_t j) const
{
    return data_[i * N_ + j];
}

DenseSquareMatrixDouble
DenseSquareMatrixDouble::operator+(const DenseSquareMatrixDouble& other) const
{
    if (N_ != other.N_)
        throw std::runtime_error("Error: Matrix dimention mismatch (+)");

    DenseSquareMatrixDouble result(N_);

    for (std::size_t i = 0; i < N_ * N_; ++i)
        result.data_[i] = data_[i] + other.data_[i];

    return result;
}

DenseSquareMatrixDouble
DenseSquareMatrixDouble::operator-(const DenseSquareMatrixDouble& other) const
{
    if (N_ != other.N_)
        throw std::runtime_error("Error: Matrix dimention mismatch (-)");

    DenseSquareMatrixDouble result(N_);

    for (std::size_t i = 0; i < N_ * N_; ++i)
        result.data_[i] = data_[i] - other.data_[i];

    return result;
}

DenseSquareMatrixDouble
DenseSquareMatrixDouble::operator*(const DenseSquareMatrixDouble& other) const
{
    if (N_ != other.N_)
        throw std::runtime_error("Error: Matrix dimention mismatch (*)");

    DenseSquareMatrixDouble result(N_);
    for (std::size_t i = 0; i < N_; ++i)
    {
        for (std::size_t k = 0; k < N_; ++k)
        {
            double aik = (*this)(i, k);

            for (std::size_t j = 0; j < N_; ++j)
            {
                result(i, j) += aik * other(k, j);
            }
        }
    }

    return result;
}


DenseSquareMatrixDouble
DenseSquareMatrixDouble::operator*(double scalar) const
{
    DenseSquareMatrixDouble result(N_);

    for (std::size_t i = 0; i < N_ * N_; ++i)
        result.data_[i] = data_[i] * scalar;

    return result;
}

VectorDouble
DenseSquareMatrixDouble::operator*(const VectorDouble& x) const
{
    if (x.size() != N_)
        throw std::runtime_error("Error: Matrix-vector dimention mismatch (*)");

    VectorDouble result(N_);

    for (std::size_t i = 0; i < N_; ++i)
    {
        double sum = 0.0;

        for (std::size_t j = 0; j < N_; ++j)
        {
            sum += (*this)(i, j) * x[j];
        }

        result[i] = sum;
    }

    return result;
}
