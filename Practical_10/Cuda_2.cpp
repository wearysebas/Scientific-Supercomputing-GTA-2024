#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to initialize matrices
__global__ void initialize_matrices(float *A, float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Flattened index
    int totalSize = N * N; // Total elements in the matrix

    if (idx < totalSize) {
        A[idx] = idx * 0.1f;      // Example initialization for A
        B[idx] = idx * 0.2f;      // Example initialization for B
    }
}

int main() {
    const int N = 5; // Define the size of the matrix (N x N)
    const int totalSize = N * N;

    // Allocate unified memory for matrices A and B
    float *A, *B;
    cudaMallocManaged(&A, totalSize * sizeof(float));
    cudaMallocManaged(&B, totalSize * sizeof(float));

    // Define the number of threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to initialize the matrices
    initialize_matrices<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Print the matrices (row-major order)
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", A[i * N + j]); // Access A in row-major order
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", B[i * N + j]); // Access B in row-major order
        }
        printf("\n");
    }

    // Free the unified memory
    cudaFree(A);
    cudaFree(B);

    return 0;
}
