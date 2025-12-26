#ifndef TENSOR_CUDA_HPP
#define TENSOR_CUDA_HPP

#include <cuda_runtime.h>
#include <cuda.h>
#include "tensor.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

/**
 * CUDA kernel for element-wise scalar product
 * Multiplies each element of the input tensor by a scalar value
 * 
 * @param input Pointer to input tensor data on device
 * @param output Pointer to output tensor data on device
 * @param scalar Scalar value to multiply by
 * @param size Total number of elements in the tensor
 */
__global__ void scalarProductKernel(const float* input, float* output, float scalar, size_t size);

/**
 * CUDA kernel for element-wise addition
 * Adds corresponding elements of two tensors
 * 
 * @param input1 Pointer to first input tensor data on device
 * @param input2 Pointer to second input tensor data on device
 * @param output Pointer to output tensor data on device
 * @param size Total number of elements in each tensor
 */
__global__ void elementWiseAddKernel(const float* input1, const float* input2, float* output, size_t size);

/**
 * CUDA kernel for matrix multiplication
 * Multiplies two 2D tensors: A[h, k] * B[k, l] = C[h, l]
 * 
 * @param A Pointer to first matrix data on device (shape: h x k)
 * @param B Pointer to second matrix data on device (shape: k x l)
 * @param C Pointer to output matrix data on device (shape: h x l)
 * @param h Number of rows in matrix A and C
 * @param k Number of columns in matrix A and rows in matrix B
 * @param l Number of columns in matrix B and C
 */
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int h, int k, int l);

// ============================================================================
// Host Wrapper Function Declarations
// ============================================================================

/**
 * CUDA wrapper function for element-wise scalar product
 * Allocates device memory, copies data, launches kernel, and copies result back
 * 
 * @param input Input tensor
 * @param scalar Scalar value to multiply by
 * @return New tensor with result of scalar product
 */
Tensor tensorScalarProductCUDA(const Tensor& input, float scalar);

/**
 * CUDA wrapper function for element-wise addition
 * Allocates device memory, copies data, launches kernel, and copies result back
 * 
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @return New tensor with result of element-wise addition
 */
Tensor tensorElementWiseAddCUDA(const Tensor& input1, const Tensor& input2);

/**
 * CUDA wrapper function for matrix multiplication
 * Multiplies two 2D tensors: A[h, k] * B[k, l] = C[h, l]
 * Allocates device memory, copies data, launches kernel, and copies result back
 * 
 * @param A First matrix tensor (shape: {h, k})
 * @param B Second matrix tensor (shape: {k, l})
 * @return New tensor with result of matrix multiplication (shape: {h, l})
 */
Tensor tensorMatrixMultiplyCUDA(const Tensor& A, const Tensor& B);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate optimal block size for CUDA kernel launch
 * 
 * @param totalElements Total number of elements to process
 * @param threadsPerBlock Number of threads per block
 * @return Number of blocks needed
 */
inline int calculateBlocks(size_t totalElements, int threadsPerBlock = 256) {
    return (totalElements + threadsPerBlock - 1) / threadsPerBlock;
}

#endif // TENSOR_CUDA_HPP

