#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int length;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &length);

    std::cout << "Proceso " << rank
              << " de " << size
              << " ejecutándose en " << hostname
              << std::endl;

    MPI_Finalize();
    return 0;
}
