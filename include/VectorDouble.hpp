#pragma once
#include <cstddef>
#include <memory>

// FIXME: here is a double vector, could be extended to numerical templates
class VectorDouble {
public:
    explicit VectorDouble(std::size_t vol);
    
    // basic operation
    VectorDouble(const VectorDouble& other); // Vector b = a
    VectorDouble& operator=(const VectorDouble& other); // b = a
    VectorDouble(VectorDouble&& other) noexcept; // Vector c = a + b
    VectorDouble& operator=(VectorDouble&& other) noexcept; // c = a + b
    ~VectorDouble() = default;

    std::size_t size() const noexcept;

    // element access: v(i)
    double& operator[](std::size_t i);
    const double& operator[](std::size_t i) const;

    // algebra
    VectorDouble operator+(const VectorDouble& other) const;
    VectorDouble operator-(const VectorDouble& other) const;
    VectorDouble operator*(double scalar) const;

    // norms
    double norm_n(int n) const;
    double normInf() const;

private:
    std::size_t vol_;
    std::unique_ptr<double[]> data_;
};
