#include <mpi.h>   // MPI header for C++
#include <iostream> // Standard C++ I/O

int main(int argc, char *argv[]) {
    int my_rank, size;

    // Initialize MPI and create the communicator MPI_COMM_WORLD
    MPI_Init(&argc, &argv);

    // Get the number of processes in the communicator (total size)
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the calling process within the communicator (my rank)
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Print "Hello world" from each process using C++ output
    std::cout << "Hello world from rank " << my_rank << " of " << size << std::endl;

    // Finalize MPI (clean up and shut down)
    MPI_Finalize();

    return 0;
}
