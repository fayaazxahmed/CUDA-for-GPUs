#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

// Helper function to check if two floats are approximately equal
bool floatEquals(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
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

// Test 1: Default constructor
void testDefaultConstructor() {
    Tensor t;
    assert(t.getOrder() == 0);
    assert(t.getTotalSize() == 0);
    assert(t.getData() == nullptr);
    std::cout << "Default constructor creates empty tensor" << std::endl;
}

// Test 2: Constructor with shape
void testShapeConstructor() {
    Tensor t({3, 4});
    assert(t.getOrder() == 2);
    assert(t.getShape()[0] == 3);
    assert(t.getShape()[1] == 4);
    assert(t.getTotalSize() == 12);
    
    // Check all elements are zero-initialized
    for (size_t i = 0; i < t.getTotalSize(); ++i) {
        assert(floatEquals(t[i], 0.0f));
    }
    std::cout << "Shape constructor creates tensor with correct dimensions" << std::endl;
}

// Test 3: Constructor with shape and initial data
void testShapeDataConstructor() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({2, 2}, data);
    assert(t.getOrder() == 2);
    assert(t.getShape()[0] == 2);
    assert(t.getShape()[1] == 2);
    assert(floatEquals(t[0], 1.0f));
    assert(floatEquals(t[1], 2.0f));
    assert(floatEquals(t[2], 3.0f));
    assert(floatEquals(t[3], 4.0f));
    std::cout << "Shape+data constructor initializes tensor correctly" << std::endl;
}

// Test 4: Copy constructor
void testCopyConstructor() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t1({2, 2}, data);
    Tensor t2(t1);
    
    assert(t2.getOrder() == t1.getOrder());
    assert(t2.getShape() == t1.getShape());
    assert(t2.getTotalSize() == t1.getTotalSize());
    for (size_t i = 0; i < t1.getTotalSize(); ++i) {
        assert(floatEquals(t1[i], t2[i]));
    }
    
    // Modify t2 and verify t1 is unchanged (deep copy)
    t2[0] = 99.0f;
    assert(floatEquals(t1[0], 1.0f));
    assert(floatEquals(t2[0], 99.0f));
    std::cout << "Copy constructor performs deep copy" << std::endl;
}

// Test 5: Move constructor
void testMoveConstructor() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t1({2, 2}, data);
    float* original_data = t1.getData();
    Tensor t2(std::move(t1));
    
    assert(t2.getOrder() == 2);
    assert(t2.getTotalSize() == 4);
    assert(t2.getData() == original_data);  // Should point to same data
    
    // t1 should be in moved-from state
    assert(t1.getOrder() == 0);
    assert(t1.getData() == nullptr);
    std::cout << "Move constructor transfers ownership correctly" << std::endl;
}

// Test 6: Copy assignment operator
void testCopyAssignment() {
    std::vector<float> data1 = {1.0f, 2.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor t1({1, 2}, data1);
    Tensor t2({2, 2}, data2);
    
    t2 = t1;
    assert(t2.getOrder() == 1);
    assert(t2.getShape()[0] == 2);
    assert(floatEquals(t2[0], 1.0f));
    assert(floatEquals(t2[1], 2.0f));
    std::cout << "Copy assignment operator works correctly" << std::endl;
}

// Test 7: Indexing operator []
void testIndexingOperator() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t({2, 3}, data);
    
    // Test reading
    assert(floatEquals(t[0], 1.0f));
    assert(floatEquals(t[3], 4.0f));
    assert(floatEquals(t[5], 6.0f));
    
    // Test writing
    t[0] = 10.0f;
    assert(floatEquals(t[0], 10.0f));
    std::cout << "Indexing operator [] works correctly" << std::endl;
}

// Test 8: Multi-dimensional indexing with at()
void testAtMethod() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({2, 2}, data);
    
    // Test reading
    assert(floatEquals(t.at({0, 0}), 1.0f));
    assert(floatEquals(t.at({0, 1}), 2.0f));
    assert(floatEquals(t.at({1, 0}), 3.0f));
    assert(floatEquals(t.at({1, 1}), 4.0f));
    
    // Test writing
    t.at({0, 0}) = 99.0f;
    assert(floatEquals(t.at({0, 0}), 99.0f));
    std::cout << "Multi-dimensional indexing with at() works correctly" << std::endl;
}

// Test 9: Scalar product
void testScalarProduct() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({2, 2}, data);
    
    Tensor result = t.scalarProduct(2.5f);
    assert(floatEquals(result[0], 2.5f));
    assert(floatEquals(result[1], 5.0f));
    assert(floatEquals(result[2], 7.5f));
    assert(floatEquals(result[3], 10.0f));
    
    // Original tensor should be unchanged
    assert(floatEquals(t[0], 1.0f));
    
    // Test operator* overload
    Tensor result2 = t * 3.0f;
    assert(floatEquals(result2[0], 3.0f));
    assert(floatEquals(result2[1], 6.0f));
    std::cout << "Scalar product works correctly" << std::endl;
}

// Test 10: Element-wise addition
void testElementWiseAdd() {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor t1({2, 2}, data1);
    Tensor t2({2, 2}, data2);
    
    Tensor result = t1.elementWiseAdd(t2);
    assert(floatEquals(result[0], 6.0f));
    assert(floatEquals(result[1], 8.0f));
    assert(floatEquals(result[2], 10.0f));
    assert(floatEquals(result[3], 12.0f));
    
    // Original tensors should be unchanged
    assert(floatEquals(t1[0], 1.0f));
    assert(floatEquals(t2[0], 5.0f));
    
    // Test operator+ overload
    Tensor result2 = t1 + t2;
    assert(floatEquals(result2[0], 6.0f));
    std::cout << "Element-wise addition works correctly" << std::endl;
}

// Test 11: Matrix multiplication
void testMatrixMultiply() {
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
    
    // Expected result C = A * B: 2x2
    // [58,  64 ]
    // [139, 154]
    // Where:
    // C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    
    Tensor C = A.matrixMultiply(B);
    assert(C.getOrder() == 2);
    assert(C.getShape()[0] == 2);
    assert(C.getShape()[1] == 2);
    
    assert(floatEquals(C.at({0, 0}), 58.0f));
    assert(floatEquals(C.at({0, 1}), 64.0f));
    assert(floatEquals(C.at({1, 0}), 139.0f));
    assert(floatEquals(C.at({1, 1}), 154.0f));
    
    // Test operator* overload
    Tensor C2 = A * B;
    assert(floatEquals(C2.at({0, 0}), 58.0f));
    
    std::cout << "Matrix multiplication works correctly" << std::endl;
}

// Test 12: 1D tensor operations
void test1DTensor() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    Tensor t({3}, data);
    
    assert(t.getOrder() == 1);
    assert(t.getShape()[0] == 3);
    assert(t.getTotalSize() == 3);
    
    Tensor scaled = t * 2.0f;
    assert(floatEquals(scaled[0], 2.0f));
    assert(floatEquals(scaled[1], 4.0f));
    assert(floatEquals(scaled[2], 6.0f));
    
    std::vector<float> data2 = {4.0f, 5.0f, 6.0f};
    Tensor t2({3}, data2);
    Tensor sum = t + t2;
    assert(floatEquals(sum[0], 5.0f));
    assert(floatEquals(sum[1], 7.0f));
    assert(floatEquals(sum[2], 9.0f));
    
    std::cout << "1D tensor operations work correctly" << std::endl;
}

// Test 13: Error handling - incompatible shapes for element-wise addition
void testErrorHandlingIncompatibleShapes() {
    Tensor t1({2, 3});
    Tensor t2({3, 2});
    
    bool exceptionThrown = false;
    try {
        Tensor result = t1 + t2;
    } catch (const std::invalid_argument& e) {
        exceptionThrown = true;
        std::cout << "  Caught expected exception: " << e.what() << std::endl;
    }
    assert(exceptionThrown);
    std::cout << "Error handling for incompatible shapes works" << std::endl;
}

// Test 14: Error handling - incompatible dimensions for matrix multiplication
void testErrorHandlingMatrixMult() {
    Tensor t1({2, 3});
    Tensor t2({2, 4});  // Incompatible: should be 3x4
    
    bool exceptionThrown = false;
    try {
        Tensor result = t1 * t2;
    } catch (const std::invalid_argument& e) {
        exceptionThrown = true;
        std::cout << "  Caught expected exception: " << e.what() << std::endl;
    }
    assert(exceptionThrown);
    std::cout << "Error handling for incompatible matrix dimensions works" << std::endl;
}

// Test 15: Print function (visual test)
void testPrint() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({2, 2}, data);
    
    std::cout << "Print output:" << std::endl;
    t.print();
    std::cout << "Print function executed successfully" << std::endl;
}

// Test 16: Complex combination of operations
void testComplexOperations() {
    // Create two matrices
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor A({2, 2}, data1);
    Tensor B({2, 2}, data2);
    
    // Compute: (A + B) * 2.0
    Tensor C = (A + B) * 2.0f;
    
    // Expected: A+B = [6, 8, 10, 12], then * 2 = [12, 16, 20, 24]
    assert(floatEquals(C[0], 12.0f));
    assert(floatEquals(C[1], 16.0f));
    assert(floatEquals(C[2], 20.0f));
    assert(floatEquals(C[3], 24.0f));
    
    std::cout << "Complex operations work correctly" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Tensor Class Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Run all tests
    runTest("Default Constructor", testDefaultConstructor);
    runTest("Shape Constructor", testShapeConstructor);
    runTest("Shape+Data Constructor", testShapeDataConstructor);
    runTest("Copy Constructor", testCopyConstructor);
    runTest("Move Constructor", testMoveConstructor);
    runTest("Copy Assignment", testCopyAssignment);
    runTest("Indexing Operator", testIndexingOperator);
    runTest("Multi-dimensional Indexing", testAtMethod);
    runTest("Scalar Product", testScalarProduct);
    runTest("Element-wise Addition", testElementWiseAdd);
    runTest("Matrix Multiplication", testMatrixMultiply);
    runTest("1D Tensor Operations", test1DTensor);
    runTest("Error Handling - Incompatible Shapes", testErrorHandlingIncompatibleShapes);
    runTest("Error Handling - Matrix Multiplication", testErrorHandlingMatrixMult);
    runTest("Print Function", testPrint);
    runTest("Complex Operations", testComplexOperations);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

