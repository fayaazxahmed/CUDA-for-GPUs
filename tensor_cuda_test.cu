#include "tensor_cuda.hpp"
#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iomanip>

// Helper function to check if two floats are approximately equal
bool floatEquals(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Helper function to check if two tensors are approximately equal
bool tensorsEqual(const Tensor& t1, const Tensor& t2, float epsilon = 1e-5f) {
    if (t1.getShape() != t2.getShape()) {
        return false;
    }
    for (size_t i = 0; i < t1.getTotalSize(); ++i) {
        if (!floatEquals(t1[i], t2[i], epsilon)) {
            return false;
        }
    }
    return true;
}

// Helper function to print tensor comparison
void printComparison(const Tensor& cpu_result, const Tensor& cuda_result, const std::string& testName) {
    std::cout << "  CPU result shape: [";
    for (size_t i = 0; i < cpu_result.getShape().size(); ++i) {
        std::cout << cpu_result.getShape()[i];
        if (i < cpu_result.getShape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  CUDA result shape: [";
    for (size_t i = 0; i < cuda_result.getShape().size(); ++i) {
        std::cout << cuda_result.getShape()[i];
        if (i < cuda_result.getShape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    if (tensorsEqual(cpu_result, cuda_result)) {
        std::cout << "  ✓ Results match!" << std::endl;
    } else {
        std::cout << "  ✗ Results do NOT match!" << std::endl;
        std::cout << "  First few elements:" << std::endl;
        size_t printCount = std::min((size_t)5, cpu_result.getTotalSize());
        for (size_t i = 0; i < printCount; ++i) {
            std::cout << "    [" << i << "] CPU: " << cpu_result[i] 
                      << ", CUDA: " << cuda_result[i] << std::endl;
        }
    }
}

// Timing helper
double getTime() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// Test function wrapper
void runTest(const std::string& testName, void (*testFunc)()) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running test: " << testName << std::endl;
    std::cout << "========================================" << std::endl;
    try {
        testFunc();
        std::cout << "✓ Test PASSED: " << testName << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Test FAILED: " << testName << std::endl;
        std::cout << "  Error: " << e.what() << std::endl;
    }
}

// Test 1: CUDA Device Check
void testCUDADevice() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "ERROR: No CUDA devices found or CUDA error occurred!" << std::endl;
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "  Device Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
}

// Test 2: Scalar Product CUDA
void testScalarProductCUDA() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor input({2, 3}, data);
    float scalar = 2.5f;
    
    // CPU computation
    double cpu_start = getTime();
    Tensor cpu_result = input.scalarProduct(scalar);
    double cpu_end = getTime();
    double cpu_time = cpu_end - cpu_start;
    
    // CUDA computation
    double cuda_start = getTime();
    Tensor cuda_result = tensorScalarProductCUDA(input, scalar);
    double cuda_end = getTime();
    double cuda_time = cuda_end - cuda_start;
    
    // Verify results match
    assert(tensorsEqual(cpu_result, cuda_result));
    
    std::cout << "  Scalar: " << scalar << std::endl;
    printComparison(cpu_result, cuda_result, "Scalar Product");
    std::cout << "  CPU time: " << std::fixed << std::setprecision(6) << cpu_time * 1000 << " ms" << std::endl;
    std::cout << "  CUDA time: " << cuda_time * 1000 << " ms" << std::endl;
    if (cpu_time > 0) {
        std::cout << "  Speedup: " << std::setprecision(2) << cpu_time / cuda_time << "x" << std::endl;
    }
}

// Test 3: Element-wise Addition CUDA
void testElementWiseAddCUDA() {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor input1({2, 2}, data1);
    Tensor input2({2, 2}, data2);
    
    // CPU computation
    double cpu_start = getTime();
    Tensor cpu_result = input1.elementWiseAdd(input2);
    double cpu_end = getTime();
    double cpu_time = cpu_end - cpu_start;
    
    // CUDA computation
    double cuda_start = getTime();
    Tensor cuda_result = tensorElementWiseAddCUDA(input1, input2);
    double cuda_end = getTime();
    double cuda_time = cuda_end - cuda_start;
    
    // Verify results match
    assert(tensorsEqual(cpu_result, cuda_result));
    
    printComparison(cpu_result, cuda_result, "Element-wise Addition");
    std::cout << "  CPU time: " << std::fixed << std::setprecision(6) << cpu_time * 1000 << " ms" << std::endl;
    std::cout << "  CUDA time: " << cuda_time * 1000 << " ms" << std::endl;
    if (cpu_time > 0) {
        std::cout << "  Speedup: " << std::setprecision(2) << cpu_time / cuda_time << "x" << std::endl;
    }
}

// Test 4: Matrix Multiplication CUDA
void testMatrixMultiplyCUDA() {
    // Create matrix A: 2x3
    // [1, 2, 3]
    // [4, 5, 6]
    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor A({2, 3}, dataA);
    
    // Create matrix B: 3x2
    // [7,  8 ]
    // [9,  10]
    // [11, 12]
    std::vector<float> dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    Tensor B({3, 2}, dataB);
    
    // CPU computation
    double cpu_start = getTime();
    Tensor cpu_result = A.matrixMultiply(B);
    double cpu_end = getTime();
    double cpu_time = cpu_end - cpu_start;
    
    // CUDA computation
    double cuda_start = getTime();
    Tensor cuda_result = tensorMatrixMultiplyCUDA(A, B);
    double cuda_end = getTime();
    double cuda_time = cuda_end - cuda_start;
    
    // Verify results match
    assert(tensorsEqual(cpu_result, cuda_result));
    
    printComparison(cpu_result, cuda_result, "Matrix Multiplication");
    std::cout << "  CPU time: " << std::fixed << std::setprecision(6) << cpu_time * 1000 << " ms" << std::endl;
    std::cout << "  CUDA time: " << cuda_time * 1000 << " ms" << std::endl;
    if (cpu_time > 0) {
        std::cout << "  Speedup: " << std::setprecision(2) << cpu_time / cuda_time << "x" << std::endl;
    }
}

// Test 5: Large Tensor Scalar Product
void testLargeScalarProduct() {
    const int size = 1000000;  // 1 million elements
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor input({size}, data);
    float scalar = 3.14f;
    
    std::cout << "  Testing with " << size << " elements" << std::endl;
    
    // CPU computation
    double cpu_start = getTime();
    Tensor cpu_result = input.scalarProduct(scalar);
    double cpu_end = getTime();
    double cpu_time = cpu_end - cpu_start;
    
    // CUDA computation
    double cuda_start = getTime();
    Tensor cuda_result = tensorScalarProductCUDA(input, scalar);
    double cuda_end = getTime();
    double cuda_time = cuda_end - cuda_start;
    
    // Verify results match (check first and last few elements)
    assert(floatEquals(cpu_result[0], cuda_result[0]));
    assert(floatEquals(cpu_result[size-1], cuda_result[size-1]));
    
    std::cout << "  CPU time: " << std::fixed << std::setprecision(6) << cpu_time * 1000 << " ms" << std::endl;
    std::cout << "  CUDA time: " << cuda_time * 1000 << " ms" << std::endl;
    if (cpu_time > 0) {
        std::cout << "  Speedup: " << std::setprecision(2) << cpu_time / cuda_time << "x" << std::endl;
    }
}

// Test 6: Large Matrix Multiplication
void testLargeMatrixMultiply() {
    const int h = 256;
    const int k = 512;
    const int l = 256;
    
    std::vector<float> dataA(h * k);
    std::vector<float> dataB(k * l);
    
    // Initialize with some values
    for (int i = 0; i < h * k; ++i) {
        dataA[i] = static_cast<float>(i % 100) / 10.0f;
    }
    for (int i = 0; i < k * l; ++i) {
        dataB[i] = static_cast<float>(i % 100) / 10.0f;
    }
    
    Tensor A({h, k}, dataA);
    Tensor B({k, l}, dataB);
    
    std::cout << "  Testing matrix multiplication: [" << h << ", " << k << "] * [" << k << ", " << l << "]" << std::endl;
    
    // CPU computation
    double cpu_start = getTime();
    Tensor cpu_result = A.matrixMultiply(B);
    double cpu_end = getTime();
    double cpu_time = cpu_end - cpu_start;
    
    // CUDA computation
    double cuda_start = getTime();
    Tensor cuda_result = tensorMatrixMultiplyCUDA(A, B);
    double cuda_end = getTime();
    double cuda_time = cuda_end - cuda_start;
    
    // Verify results match (check a few elements)
    assert(floatEquals(cpu_result.at({0, 0}), cuda_result.at({0, 0})));
    assert(floatEquals(cpu_result.at({h-1, l-1}), cuda_result.at({h-1, l-1})));
    
    std::cout << "  CPU time: " << std::fixed << std::setprecision(6) << cpu_time * 1000 << " ms" << std::endl;
    std::cout << "  CUDA time: " << cuda_time * 1000 << " ms" << std::endl;
    if (cpu_time > 0) {
        std::cout << "  Speedup: " << std::setprecision(2) << cpu_time / cuda_time << "x" << std::endl;
    }
}

// Test 7: Error Handling - Incompatible Shapes
void testErrorHandlingIncompatibleShapes() {
    Tensor t1({2, 3});
    Tensor t2({3, 2});
    
    bool exceptionThrown = false;
    try {
        Tensor result = tensorElementWiseAddCUDA(t1, t2);
    } catch (const std::invalid_argument& e) {
        exceptionThrown = true;
        std::cout << "  Caught expected exception: " << e.what() << std::endl;
    }
    assert(exceptionThrown);
    std::cout << "Error handling for incompatible shapes works" << std::endl;
}

// Test 8: Error Handling - Matrix Multiplication Dimensions
void testErrorHandlingMatrixMult() {
    Tensor t1({2, 3});
    Tensor t2({2, 4});  // Incompatible: should be 3x4
    
    bool exceptionThrown = false;
    try {
        Tensor result = tensorMatrixMultiplyCUDA(t1, t2);
    } catch (const std::invalid_argument& e) {
        exceptionThrown = true;
        std::cout << "  Caught expected exception: " << e.what() << std::endl;
    }
    assert(exceptionThrown);
    std::cout << "Error handling for incompatible matrix dimensions works" << std::endl;
}

// Test 9: 1D Tensor Operations
void test1DTensorCUDA() {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> data2 = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    
    Tensor t1({5}, data1);
    Tensor t2({5}, data2);
    
    // Test scalar product
    Tensor scaled = tensorScalarProductCUDA(t1, 2.0f);
    assert(floatEquals(scaled[0], 2.0f));
    assert(floatEquals(scaled[4], 10.0f));
    
    // Test element-wise addition
    Tensor sum = tensorElementWiseAddCUDA(t1, t2);
    assert(floatEquals(sum[0], 7.0f));
    assert(floatEquals(sum[4], 15.0f));
    
    std::cout << "1D tensor CUDA operations work correctly" << std::endl;
}

// Test 10: Multiple Operations Chain
void testMultipleOperations() {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor A({2, 2}, data1);
    Tensor B({2, 2}, data2);
    
    // Compute: (A + B) * 2.0 using CUDA
    Tensor sum = tensorElementWiseAddCUDA(A, B);
    Tensor result = tensorScalarProductCUDA(sum, 2.0f);
    
    // Expected: A+B = [6, 8, 10, 12], then * 2 = [12, 16, 20, 24]
    assert(floatEquals(result[0], 12.0f));
    assert(floatEquals(result[1], 16.0f));
    assert(floatEquals(result[2], 20.0f));
    assert(floatEquals(result[3], 24.0f));
    
    std::cout << "Multiple CUDA operations chained correctly" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Tensor CUDA Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Check CUDA device first
    testCUDADevice();
    
    // Run all tests
    runTest("Scalar Product CUDA", testScalarProductCUDA);
    runTest("Element-wise Addition CUDA", testElementWiseAddCUDA);
    runTest("Matrix Multiplication CUDA", testMatrixMultiplyCUDA);
    runTest("Large Tensor Scalar Product", testLargeScalarProduct);
    runTest("Large Matrix Multiplication", testLargeMatrixMultiply);
    runTest("1D Tensor CUDA Operations", test1DTensorCUDA);
    runTest("Multiple Operations Chain", testMultipleOperations);
    runTest("Error Handling - Incompatible Shapes", testErrorHandlingIncompatibleShapes);
    runTest("Error Handling - Matrix Multiplication", testErrorHandlingMatrixMult);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All CUDA tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

