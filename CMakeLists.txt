
cmake_minimum_required(VERSION 3.15)
project(knn2 LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Version secuencial
add_executable(cv
    main_secuencial.cc
    crossValidation.cc
    mnist.cc
    knn.cc
    sample.cc
)
target_include_directories(cv PRIVATE ${PROJECT_SOURCE_DIR})

# Version MPI estatica
add_executable(cv_mpi_estatico
    main_mpi_estatico.cc
    crossValidation.cc
    mnist.cc
    knn.cc
    sample.cc
)
target_include_directories(cv_mpi_estatico PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(cv_mpi_estatico PRIVATE MPI::MPI_CXX)

# Version MPI dinamica
add_executable(cv_mpi_dinamico
    main_mpi_dinamico.cc
    crossValidation.cc
    mnist.cc
    knn.cc
    sample.cc
)
target_include_directories(cv_mpi_dinamico PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(cv_mpi_dinamico PRIVATE MPI::MPI_CXX)

# Tests
include(CTest)

if(BUILD_TESTING)
    add_executable(knn2_tests
        tests.cc
        sample.cc
        knn.cc
        mnist.cc
        crossValidation.cc
    )
    add_test(NAME knn2_tests COMMAND knn2_tests)
endif()
