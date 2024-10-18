#include <mpi.h>
#include <iostream>   // for std::cout
#include <cstdlib>    // for rand(), srand()
#include <ctime>      // for time()

int main(int argc, char *argv[]) {
    int my_rank, size;
    int numpasses;
    double parcel;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Seed the random number generator
    srand(time(NULL) + my_rank);

    // On rank 0: Set the number of passes and initial parcel value
    if (my_rank == 0) {
        numpasses = rand() % 101;  // Random number between 0 and 100
        parcel = (double)rand() / RAND_MAX * 2.0 - 1.0;  // Random number between -1 and +1
        std::cout << "Initial numpasses: " << numpasses << ", Initial parcel: " << parcel << std::endl;
    }

    // Broadcast the numpasses to all processes
    MPI_Bcast(&numpasses, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Begin the "Pass the Parcel" loop
    int current_pass = 0;
    while (current_pass < numpasses) {
        if (my_rank == current_pass % size) {
            // This rank holds the parcel
            if (current_pass == 0) {
                // The first pass, only rank 0 starts with the parcel
                if (my_rank == 0) {
                    MPI_Send(&parcel, 1, MPI_DOUBLE, (my_rank + 1) % size, 0, MPI_COMM_WORLD);
                    std::cout << "Rank " << my_rank << " sent the parcel: " << parcel << " to rank " << (my_rank + 1) % size << std::endl;
                }
            } else {
                // Receive the parcel
                MPI_Recv(&parcel, 1, MPI_DOUBLE, (my_rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "Rank " << my_rank << " received the parcel: " << parcel << " from rank " << (my_rank - 1 + size) % size << std::endl;
                
                // Modify the parcel (unwrap a layer)
                double modification = (double)rand() / RAND_MAX * 2.0 - 1.0;  // Random number between -1 and +1
                parcel += modification;
                std::cout << "Rank " << my_rank << " modified the parcel by " << modification << ", new parcel: " << parcel << std::endl;
                
                // Send the parcel to the next rank
                MPI_Send(&parcel, 1, MPI_DOUBLE, (my_rank + 1) % size, 0, MPI_COMM_WORLD);
                std::cout << "Rank " << my_rank << " sent the parcel: " << parcel << " to rank " << (my_rank + 1) % size << std::endl;
            }
        }
        current_pass++;
    }

    // Final pass: Print the final contents of the parcel on the process that holds it
    if (my_rank == (numpasses % size)) {
        MPI_Recv(&parcel, 1, MPI_DOUBLE, (my_rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Final parcel after " << numpasses << " passes is: " << parcel << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
