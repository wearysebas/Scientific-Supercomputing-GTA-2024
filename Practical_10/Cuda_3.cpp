#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to perform matrix addition
__global__ void matrix_add(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Flattened index
    int totalSize = N * N;

    if (idx < totalSize) {
        A[idx] = B[idx] + C[idx]; // Perform addition
    }
}

// Kernel to initialize matrices
__global__ void initialize_matrices(float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Flattened index
    int totalSize = N * N;

    if (idx < totalSize) {
        B[idx] = 1.0f; // Initialize B to 1.0
        C[idx] = 2.0f; // Initialize C to 2.0
    }
}

// Host function to verify the result and calculate max error
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
    const int N = 5; // Size of the matrix (N x N)
    const int totalSize = N * N;

    // Allocate unified memory for matrices
    float *A, *B, *C;
    cudaMallocManaged(&A, totalSize * sizeof(float));
    cudaMallocManaged(&B, totalSize * sizeof(float));
    cudaMallocManaged(&C, totalSize * sizeof(float));

    // Define the number of threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize matrices B and C
    initialize_matrices<<<blocksPerGrid, threadsPerBlock>>>(B, C, N);
    cudaDeviceSynchronize(); // Ensure initialization is complete

    // Perform matrix addition A = B + C
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize(); // Ensure addition is complete

    // Verify the result and compute maximum error
    float max_error = verify_and_compute_error(A, N);

    // Print matrices and results
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\nMaximum error (delta): %f\n", max_error);

    // Free unified memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
