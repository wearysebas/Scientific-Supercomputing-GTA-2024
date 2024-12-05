#include <stdio.h>

// GPU Kernel
__global__ void hello_from_gpu() {
    printf("Hello, World! from GPU thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch the kernel with 1 block of 10 threads
    hello_from_gpu<<<1, 10>>>();

    // Synchronize the device to ensure all output is flushed before exiting
    cudaDeviceSynchronize();

    return 0;
}
