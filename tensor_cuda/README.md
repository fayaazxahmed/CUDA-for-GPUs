# Tensor and Tensor_CUDA Programs

This folder contains solutions for the tensor and tensor_cuda assignments. It requires Linux OS and NVIDIA GPU installed. You can do this assignment question on Colab with runtime T4, and use Terminal operations. 

## Prerequisites
- Colab with runtime type T4
- NVIDIA GPU and CUDA Toolkit (for CUDA version)
- C/C++ compiler (g++, nvcc)
- Mount to Google Drive.

## Build Instructions

### Build and Test Tensor (CPU version)
1. Command termial to Colab
2. Navigate to this folder:
   ```bash
   cd drive/MyDrive/tensor_cuda
   ```
3. Compile the CPU version:
   ```bash
   g++  tensor.cpp tensor_test.cpp -o tensor
   ```
4. Test the CPU version:
  ```bash
  ./tensor
  ```   

### Build and Test Tensor_CUDA (GPU version)
1. Ensure CUDA Toolkit is installed and `nvcc` is available in your PATH.
2. Compile the CUDA version (for compute capability 7.5):
   ```bash
   nvcc -arch=sm_75 tensor.cpp tensor_cuda.cu tensor_cuda_test.cu -o tensor_cuda
   ```
3. Test the CUDA version:
  ```bash
  ./tensor_cuda
  ```
4. Compile CPU and GPU tensor computing :
   ```bash
   nvcc -arch=sm_75 tensor.cpp tensor_cuda.cu tensor_cuda_compare.cu -o tensor_cuda_compare
   ```  
5. To test the CPU vs CUDA comparison:
  ```bash
  ./tensor_cuda_compare
  ```

## Clean Instructions

- To remove executables and object files:
  ```bash
  rm tensor tensor_cuda tensor_cuda_compare *.o
  ```


## Use makefile


1. makefile_cuda is the make file from the CUDA tensor

2. Build 
  ```bash
  make -f makefile_cuda
  ```
tensor_cuda_test.out and tensor_cuda_compare.out will be created.  

3. Run both programs

  ```bash
  make -f makefile_cuda run
  ```
4. Run both programs


Compile and build by 


## CUDA Colab Notebook (cuda_colab.ipynb)

Jupyter notebook with examples of GPU computing, from basic vector operations to advanced graphics generation.

### Notebook Contents

The notebook is structured as a progressive tutorial covering:

1. **Basic CUDA Examples**: 
   - Vector addition program demonstrating basic kernel execution
   - Dot product computation using shared memory optimization
2. **GPU Graphics Generation**: Advanced section on generating fractal visualizations using GPU acceleration
3. **Python Visualization**: Integrating CUDA-generated data with Python (NumPy, matplotlib) for visualization
4. **Performance Analysis**: Comparing CPU vs GPU implementations to demonstrate the speedup achieved through parallelization

### GPU-Accelerated Fractal Calculations

The notebook includes examples of generating Mandelbrot and Julia set fractals using GPU acceleration. Here's how the GPU is utilized for these calculations:

**Parallel Pixel Computation:**
- Each pixel in the fractal image is computed independently, making it an ideal parallel workload
- The CUDA kernel launches a 2D grid of thread blocks (typically 16×16 threads per block)
- Each thread computes one pixel, calculating the iteration count for that pixel's complex number
- For a 1024×1024 image, over 1 million threads execute simultaneously on the GPU

**Kernel Execution Pattern:**
```cuda
// Each thread handles one pixel
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

**Workflow:**
1. Host allocates device memory for the output image
2. GPU kernel launches with 2D grid/block configuration
3. Each thread computes its pixel's iteration count (Mandelbrot/Julia set formula)
4. Results are written to device memory
5. Data is copied back to host and saved to binary file
6. Python reads the binary file and visualizes using matplotlib

**Performance Benefits:**
- Thousands of pixels computed simultaneously vs. sequential CPU computation
- Achieves 100x+ speedup compared to CPU implementations
- Enables real-time generation of high-resolution fractal images

The notebook demonstrates how this GPU-accelerated data generation seamlessly integrates with Python visualization tools, showing the complete workflow from CUDA kernel development to final graphics output.

## Notes
- Make sure all required libraries and toolkits are installed and available in your environment.
- For CUDA, ensure your GPU drivers and CUDA Toolkit are properly configured.
- If you encounter issues, check your compiler and CUDA installation.
