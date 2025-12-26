#include "tensor_cuda.hpp"
#include <iostream>
#include <cstdlib>

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

/**
 * CUDA kernel for element-wise scalar product
 * Each thread processes one element
 */
__global__ void scalarProductKernel(const float* input, float* output, float scalar, size_t size) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process element if within bounds
    if (idx < size) {
        output[idx] = input[idx] * scalar;
    }
}

/**
 * CUDA kernel for element-wise addition
 * Each thread processes one element
 */
__global__ void elementWiseAddKernel(const float* input1, const float* input2, float* output, size_t size) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process element if within bounds
    if (idx < size) {
        output[idx] = input1[idx] + input2[idx];
    }
}

/**
 * CUDA kernel for matrix multiplication using shared memory optimization
 * Multiplies A[h, k] * B[k, l] = C[h, l]
 * Uses tile-based approach with shared memory for better performance
 */
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int h, int k, int l) {
    // Shared memory for tiles
    // TILE_SIZE should be chosen based on available shared memory (typically 16 or 32)
    const int TILE_SIZE = 16;
    
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Calculate row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile from A into shared memory
        int aRow = row;
        int aCol = tile * TILE_SIZE + threadIdx.x;
        if (aRow < h && aCol < k) {
            tileA[threadIdx.y][threadIdx.x] = A[aRow * k + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        int bRow = tile * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < k && bCol < l) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * l + bCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < h && col < l) {
        C[row * l + col] = sum;
    }
}

// ============================================================================
// Host Wrapper Function Implementations
// ============================================================================

/**
 * CUDA wrapper function for element-wise scalar product
 */
Tensor tensorScalarProductCUDA(const Tensor& input, float scalar) {
    // Get input tensor properties
    size_t size = input.getTotalSize();
    const float* host_input = input.getData();
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(float)));
    
    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, host_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = calculateBlocks(size, threadsPerBlock);
    
    // Launch kernel
    scalarProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar, size);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Allocate host memory for result
    std::vector<float> host_output(size);
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(host_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    // Create result tensor
    return Tensor(input.getShape(), host_output);
}

/**
 * CUDA wrapper function for element-wise addition
 */
Tensor tensorElementWiseAddCUDA(const Tensor& input1, const Tensor& input2) {
    // Validate that tensors have the same shape
    if (input1.getShape() != input2.getShape()) {
        throw std::invalid_argument("Tensors must have the same shape for element-wise addition");
    }
    
    // Get tensor properties
    size_t size = input1.getTotalSize();
    const float* host_input1 = input1.getData();
    const float* host_input2 = input2.getData();
    
    // Allocate device memory
    float* d_input1 = nullptr;
    float* d_input2 = nullptr;
    float* d_output = nullptr;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input1, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_input2, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(float)));
    
    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_input1, host_input1, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input2, host_input2, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = calculateBlocks(size, threadsPerBlock);
    
    // Launch kernel
    elementWiseAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, size);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Allocate host memory for result
    std::vector<float> host_output(size);
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(host_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input1));
    CUDA_CHECK(cudaFree(d_input2));
    CUDA_CHECK(cudaFree(d_output));
    
    // Create result tensor
    return Tensor(input1.getShape(), host_output);
}

/**
 * CUDA wrapper function for matrix multiplication
 * Multiplies A[h, k] * B[k, l] = C[h, l]
 */
Tensor tensorMatrixMultiplyCUDA(const Tensor& A, const Tensor& B) {
    // Validate that both tensors are 2D
    if (A.getOrder() != 2 || B.getOrder() != 2) {
        throw std::invalid_argument("Matrix multiplication requires both tensors to be 2D");
    }
    
    // Get matrix dimensions
    int h = A.getShape()[0];
    int k = A.getShape()[1];
    int k_other = B.getShape()[0];
    int l = B.getShape()[1];
    
    // Validate dimension compatibility
    if (k != k_other) {
        throw std::invalid_argument(
            "Matrix multiplication requires: first tensor columns (k) must equal second tensor rows (k)");
    }
    
    // Get input data pointers
    const float* host_A = A.getData();
    const float* host_B = B.getData();
    
    // Allocate device memory
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, h * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, k * l * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, h * l * sizeof(float)));
    
    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, host_A, h * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, host_B, k * l * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize output matrix to zero
    CUDA_CHECK(cudaMemset(d_C, 0, h * l * sizeof(float)));
    
    // Configure kernel launch parameters
    // Use 16x16 thread blocks for tile-based matrix multiplication
    const int TILE_SIZE = 16;
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((l + TILE_SIZE - 1) / TILE_SIZE, (h + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, h, k, l);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Allocate host memory for result
    std::vector<float> host_C(h * l);
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(host_C.data(), d_C, h * l * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Create result tensor with shape {h, l}
    return Tensor({h, l}, host_C);
}

