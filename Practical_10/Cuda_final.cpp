#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono> // For timing

// CUDA kernel to perform matrix addition
__global__ void matrix_add(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * N;

    if (idx < totalSize) {
        A[idx] = B[idx] + C[idx];
    }
}

// Kernel to initialize matrices
__global__ void initialize_matrices(float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * N;

    if (idx < totalSize) {
        B[idx] = 1.0f;
        C[idx] = 2.0f;
    }
}

// Host (CPU) function to perform matrix addition
void cpu_matrix_add(float *A, float *B, float *C, int N) {
    int totalSize = N * N;
    for (int i = 0; i < totalSize; i++) {
        A[i] = B[i] + C[i];
    }
}

// Function to verify and compute the maximum error
float verify_and_compute_error(float *A, int N) {
    float max_error = 0.0f;
    int totalSize = N * N;

    for (int i = 0; i < totalSize; i++) {
        float expected = 3.0f; // Expected result
        float error = fabs(A[i] - expected);
        if (error > max_error) {
            max_error = error;
        }
    }

    return max_error;
}

int main() {
    const int N = 1024; // Matrix size (N x N)
    const int totalSize = N * N;

    // Allocate unified memory
    float *A_gpu, *B, *C;
    float *A_cpu; // Separate array for CPU result
    cudaMallocManaged(&A_gpu, totalSize * sizeof(float));
    cudaMallocManaged(&B, totalSize * sizeof(float));
    cudaMallocManaged(&C, totalSize * sizeof(float));
    A_cpu = (float *)malloc(totalSize * sizeof(float));

    // Initialize matrices B and C
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    initialize_matrices<<<blocksPerGrid, threadsPerBlock>>>(B, C, N);
    cudaDeviceSynchronize();

    // GPU matrix addition and timing
    auto gpu_start = std::chrono::high_resolution_clock::now();
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A_gpu, B, C, N);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    // CPU matrix addition and timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matrix_add(A_cpu, B, C, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Verify GPU result and compute maximum error
    float max_error = verify_and_compute_error(A_gpu, N);

    // Print results
    printf("Matrix size: %d x %d\n", N, N);
    printf("GPU computation time: %.2f ms\n", gpu_time);
    printf("CPU computation time: %.2f ms\n", cpu_time);
    printf("Speedup (CPU vs GPU): %.2fx\n", cpu_time / gpu_time);
    printf("Maximum error (delta): %.6f\n", max_error);

    // Free memory
    cudaFree(A_gpu);
    cudaFree(B);
    cudaFree(C);
    free(A_cpu);

    return 0;
}
