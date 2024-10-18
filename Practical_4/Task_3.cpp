#include <mpi.h>
#include <iostream>  // for std::cout

//modified hello_world to print the results in order.

int main(int argc, char *argv[]) {
    int my_rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Loop over all ranks to ensure each process prints in order
    for (int i = 0; i < size; ++i) {
        // Wait for other processes to reach this point (synchronize)
        MPI_Barrier(MPI_COMM_WORLD);

        // Only the process with the current rank (i) prints its message
        if (my_rank == i) {
            std::cout << "Hello world from rank " << my_rank << " of " << size << std::endl;
        }

        // Wait for other processes before allowing the next one to print
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
