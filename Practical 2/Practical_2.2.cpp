#include <iostream>
#include <omp.h>  // OpenMP header for multithreading

int main() {
    // Define the size of the array (e.g., N = 1,000,000)
    const int N = 1000000;

    // Allocate an array of size N
    int* array = new int[N];

    // Initialize the array such that each value equals its index (1 to N)
    for (int i = 0; i < N; ++i) {
        array[i] = i + 1;  // Set array[i] = i + 1 (i.e., 1, 2, 3, ..., N)
    }

    long long sum = 0;  // Use long long to store the sum to prevent overflow

    // Calculate the sum of all elements in the array using multithreading
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += array[i];
    }

    // Print out the result of the sum
    std::cout << "The sum of integers from 1 to " << N << " is: " << sum << std::endl;

    // Free the allocated memory
    delete[] array;

    return 0;
}
