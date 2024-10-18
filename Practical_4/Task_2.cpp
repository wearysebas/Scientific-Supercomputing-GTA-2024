#include <mpi.h>
#include <iostream>  // for std::cout

int main(int argc, char *argv[]) {
    int my_rank, size, my_data;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Check if there are at least two processes for ping-pong
    if (size < 2) {
        if (my_rank == 0) {
            std::cerr << "This program requires at least two processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Set my_data to the rank of the current process
    my_data = my_rank;

    // Rank 0 sends data to rank 1 and rank 1 receives it
    if (my_rank == 0) {
        // Send my_data from rank 0 to rank 1
        MPI_Send(&my_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Rank 0 sent data: " << my_data << std::endl;

        // Receive the data back from rank 1
        MPI_Recv(&my_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Rank 0 received data back: " << my_data << std::endl;
    } else if (my_rank == 1) {
        // Receive data from rank 0
        MPI_Recv(&my_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Rank 1 received data: " << my_data << std::endl;

        // Reset my_data to rank of the current process (rank 1)
        my_data = my_rank;

        // Send the data back to rank 0
        MPI_Send(&my_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        std::cout << "Rank 1 sent data back: " << my_data << std::endl;
    }

    // Finalize MPI (clean up)
    MPI_Finalize();

    return 0;
}
