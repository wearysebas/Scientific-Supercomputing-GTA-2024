#include <iostream>
#include <omp.h>  // Include OpenMP header

int main() {
    // Begin parallel region
    #pragma omp parallel
    {
        // Get the total number of threads and the current thread number
        int total_num_threads = omp_get_num_threads();
        int my_thread_number = omp_get_thread_num();

        // Print the message from each thread
        std::cout << "Hello world from thread " << my_thread_number 
                  << " of " << total_num_threads << std::endl;
    }
    // End parallel region

    return 0;
}
