#include "VectorDouble.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

VectorDouble::VectorDouble(std::size_t vol)
    : vol_(vol), data_(std::make_unique<double[]>(vol))
{
    for (std::size_t i = 0; i < vol_; ++i)
        data_[i] = 0.0;
}

VectorDouble::VectorDouble(const VectorDouble& other)
    : vol_(other.vol_), data_(std::make_unique<double[]>(other.vol_))
{
    for (std::size_t i = 0; i < vol_; ++i)
        data_[i] = other.data_[i];
}

VectorDouble& VectorDouble::operator=(const VectorDouble& other)
{
    if (this == &other)
        return *this;

    if (vol_ != other.vol_) {
        vol_ = other.vol_;
        data_ = std::make_unique<double[]>(vol_);
    }

    for (std::size_t i = 0; i < vol_; ++i)
        data_[i] = other.data_[i];

    return *this;
}

VectorDouble::VectorDouble(VectorDouble&& other) noexcept
    : vol_(other.vol_), data_(std::move(other.data_))
{
    other.vol_ = 0;
}

VectorDouble& VectorDouble::operator=(VectorDouble&& other) noexcept
{
    if (this == &other)
        return *this;

    vol_ = other.vol_;
    data_ = std::move(other.data_);
    other.vol_ = 0;

    return *this;
}

std::size_t VectorDouble::size() const noexcept
{
    return vol_;
}

double& VectorDouble::operator[](std::size_t i)
{
    return data_[i];
}

const double& VectorDouble::operator[](std::size_t i) const
{
    return data_[i];
}

VectorDouble VectorDouble::operator+(const VectorDouble& other) const
{
    if (vol_ != other.vol_)
        throw std::runtime_error("Error: Vector size mismatch (+)");

    VectorDouble result(vol_);
    for (std::size_t i = 0; i < vol_; ++i)
        result[i] = data_[i] + other.data_[i];

    return result;
}

VectorDouble VectorDouble::operator-(const VectorDouble& other) const
{
    if (vol_ != other.vol_)
        throw std::runtime_error("Error: Vector size mismatch (-)");

    VectorDouble result(vol_);
    for (std::size_t i = 0; i < vol_; ++i)
        result[i] = data_[i] - other.data_[i];

    return result;
}

VectorDouble VectorDouble::operator*(double scalar) const
{
    VectorDouble result(vol_);
    for (std::size_t i = 0; i < vol_; ++i)
        result[i] = data_[i] * scalar;

    return result;
}

double VectorDouble::norm_n(int n) const
{
    if (n <= 0)
        throw std::runtime_error("Error: Invalid norm parameter");

    double sum = 0.0;

    for (std::size_t i = 0; i < vol_; ++i)
        sum += std::pow(std::abs(data_[i]), n);

    return std::pow(sum, 1.0 / n);
}

double VectorDouble::normInf() const
{
    double maxVal = 0.0;

    for (std::size_t i = 0; i < vol_; ++i)
        maxVal = std::max(maxVal, std::abs(data_[i]));

    return maxVal;
}
