#include "tensor.hpp"
#include <cstring>

// Helper function to calculate total size from shape
size_t Tensor::calculateSize(const std::vector<int>& shape) const {
    size_t size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    return size;
}

// Helper function to convert multi-dimensional index to flat index
size_t Tensor::getFlatIndex(const std::vector<int>& indices) const {
    if (indices.size() != static_cast<size_t>(order)) {
        throw std::invalid_argument("Number of indices must match tensor order");
    }
    
    size_t flat_index = 0;
    size_t stride = 1;
    for (int i = order - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        flat_index += indices[i] * stride;
        stride *= shape[i];
    }
    return flat_index;
}

// Default constructor
Tensor::Tensor() : order(0), data(nullptr), total_size(0) {}

// Constructor with shape
Tensor::Tensor(const std::vector<int>& shape) 
    : order(static_cast<int>(shape.size())), shape(shape) {
    if (order < 0) {
        throw std::invalid_argument("Tensor order cannot be negative");
    }
    total_size = calculateSize(shape);
    data = new float[total_size];
    // Initialize to zero
    std::memset(data, 0, total_size * sizeof(float));
}

// Constructor with shape and initial data
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& initial_data)
    : order(static_cast<int>(shape.size())), shape(shape) {
    if (order < 0) {
        throw std::invalid_argument("Tensor order cannot be negative");
    }
    total_size = calculateSize(shape);
    if (initial_data.size() != total_size) {
        throw std::invalid_argument("Initial data size does not match tensor shape");
    }
    data = new float[total_size];
    std::memcpy(data, initial_data.data(), total_size * sizeof(float));
}

// Copy constructor
Tensor::Tensor(const Tensor& other) 
    : order(other.order), shape(other.shape), total_size(other.total_size) {
    if (other.data != nullptr) {
        data = new float[total_size];
        std::memcpy(data, other.data, total_size * sizeof(float));
    } else {
        data = nullptr;
    }
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : order(other.order), shape(std::move(other.shape)), 
      data(other.data), total_size(other.total_size) {
    other.order = 0;
    other.data = nullptr;
    other.total_size = 0;
}

// Destructor
Tensor::~Tensor() {
    delete[] data;
}

// Copy assignment operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data;
        order = other.order;
        shape = other.shape;
        total_size = other.total_size;
        if (other.data != nullptr) {
            data = new float[total_size];
            std::memcpy(data, other.data, total_size * sizeof(float));
        } else {
            data = nullptr;
        }
    }
    return *this;
}

// Move assignment operator
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data;
        order = other.order;
        shape = std::move(other.shape);
        data = other.data;
        total_size = other.total_size;
        other.order = 0;
        other.data = nullptr;
        other.total_size = 0;
    }
    return *this;
}

// Getter methods
int Tensor::getOrder() const {
    return order;
}

const std::vector<int>& Tensor::getShape() const {
    return shape;
}

size_t Tensor::getTotalSize() const {
    return total_size;
}

const float* Tensor::getData() const {
    return data;
}

float* Tensor::getData() {
    return data;
}

// Access element by flat index
float& Tensor::operator[](size_t index) {
    if (index >= total_size) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[index];
}

const float& Tensor::operator[](size_t index) const {
    if (index >= total_size) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[index];
}

// Access element by multi-dimensional indices
float& Tensor::at(const std::vector<int>& indices) {
    return data[getFlatIndex(indices)];
}

const float& Tensor::at(const std::vector<int>& indices) const {
    return data[getFlatIndex(indices)];
}

// Scalar product: multiply tensor by a scalar
Tensor Tensor::scalarProduct(float scalar) const {
    Tensor result(shape);
    for (size_t i = 0; i < total_size; ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

// Overload operator* for scalar product
Tensor Tensor::operator*(float scalar) const {
    return scalarProduct(scalar);
}

// Element-wise addition: add two tensors element-wise
Tensor Tensor::elementWiseAdd(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Tensors must have the same shape for element-wise addition");
    }
    Tensor result(shape);
    for (size_t i = 0; i < total_size; ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

// Overload operator+ for element-wise addition
Tensor Tensor::operator+(const Tensor& other) const {
    return elementWiseAdd(other);
}

// Matrix multiplication: multiply two 2D tensors of shapes {h, k} and {k, l}
Tensor Tensor::matrixMultiply(const Tensor& other) const {
    // Check if both tensors are 2D
    if (order != 2 || other.order != 2) {
        throw std::invalid_argument("Matrix multiplication requires both tensors to be 2D");
    }
    
    // Check if dimensions are compatible: {h, k} * {k, l} = {h, l}
    int h = shape[0];
    int k = shape[1];
    int k_other = other.shape[0];
    int l = other.shape[1];
    
    if (k != k_other) {
        throw std::invalid_argument(
            "Matrix multiplication requires: first tensor columns (k) must equal second tensor rows (k)");
    }
    
    // Create result tensor with shape {h, l}
    Tensor result({h, l});
    
    // Perform matrix multiplication: C[i][j] = sum(A[i][m] * B[m][j])
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < l; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < k; ++m) {
                // A[i][m] = data[i * k + m]
                // B[m][j] = other.data[m * l + j]
                sum += data[i * k + m] * other.data[m * l + j];
            }
            // C[i][j] = result.data[i * l + j]
            result.data[i * l + j] = sum;
        }
    }
    
    return result;
}

// Overload operator* for matrix multiplication
Tensor Tensor::operator*(const Tensor& other) const {
    return matrixMultiply(other);
}

// Print tensor (for debugging)
void Tensor::print() const {
    std::cout << "Tensor(order=" << order << ", shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "])" << std::endl;
    
    if (order == 0) {
        std::cout << "Empty tensor" << std::endl;
        return;
    }
    
    if (order == 1) {
        // 1D tensor
        std::cout << "[";
        for (size_t i = 0; i < total_size; ++i) {
            std::cout << data[i];
            if (i < total_size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    } else if (order == 2) {
        // 2D tensor (matrix)
        int rows = shape[0];
        int cols = shape[1];
        for (int i = 0; i < rows; ++i) {
            std::cout << "[";
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i * cols + j];
                if (j < cols - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    } else {
        // Higher dimensional tensor - print flat representation
        std::cout << "Data: [";
        for (size_t i = 0; i < total_size; ++i) {
            std::cout << data[i];
            if (i < total_size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

