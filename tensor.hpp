#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstring>

class Tensor {
private:
    int order;                      // Dimension of the tensor
    std::vector<int> shape;         // Sizes of each dimension
    float* data;                    // Array of data elements
    size_t total_size;              // Total number of elements

    // Helper function to calculate total size from shape
    size_t calculateSize(const std::vector<int>& shape) const;

    // Helper function to convert multi-dimensional index to flat index
    size_t getFlatIndex(const std::vector<int>& indices) const;

public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<float>& initial_data);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    // Destructor
    ~Tensor();

    // Assignment operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Getter methods
    int getOrder() const;
    const std::vector<int>& getShape() const;
    size_t getTotalSize() const;
    const float* getData() const;
    float* getData();

    // Access element by flat index
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    // Access element by multi-dimensional indices
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;

    // Scalar product: multiply tensor by a scalar
    Tensor scalarProduct(float scalar) const;
    Tensor operator*(float scalar) const;

    // Element-wise addition: add two tensors element-wise
    Tensor elementWiseAdd(const Tensor& other) const;
    Tensor operator+(const Tensor& other) const;

    // Matrix multiplication: multiply two 2D tensors of shapes {h, k} and {k, l}
    Tensor matrixMultiply(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    // Print tensor (for debugging)
    void print() const;
};

#endif // TENSOR_HPP

