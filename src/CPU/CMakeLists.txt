cmake_minimum_required(VERSION 3.12)
project(FOFHaloFinderParticleCompressor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)
find_package(OpenMP REQUIRED)

# Optional LIKWID marker instrumentation for per-region CPU roofline profiling
# (memory bandwidth + FLOP/s -> arithmetic intensity). Enable with:
#   cmake -DUSE_LIKWID=ON ..   (requires liblikwid + likwid-marker.h; e.g. `module load likwid`)
option(USE_LIKWID "Enable LIKWID marker instrumentation" OFF)
if(USE_LIKWID)
    find_library(LIKWID_LIB NAMES likwid)
    find_path(LIKWID_INCLUDE_DIR NAMES likwid-marker.h)
    if(NOT LIKWID_LIB OR NOT LIKWID_INCLUDE_DIR)
        message(FATAL_ERROR "USE_LIKWID=ON but liblikwid/likwid-marker.h not found; `module load likwid` or set LIKWID_LIB and LIKWID_INCLUDE_DIR.")
    endif()
    message(STATUS "LIKWID enabled: ${LIKWID_LIB} (include: ${LIKWID_INCLUDE_DIR})")
endif()

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create executable
add_executable(fofpz_cpu
    main.cpp
    fileIO.cpp
)

# Link Packages
target_include_directories(fofpz_cpu PRIVATE ${ZSTD_INCLUDE_DIRS})
target_link_libraries(fofpz_cpu PRIVATE ${ZSTD_LIBRARIES} OpenMP::OpenMP_CXX)
if(USE_LIKWID)
    target_compile_definitions(fofpz_cpu PRIVATE USE_LIKWID)
    target_include_directories(fofpz_cpu PRIVATE ${LIKWID_INCLUDE_DIR})
    target_link_libraries(fofpz_cpu PRIVATE ${LIKWID_LIB})
endif()

# Enable optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(fofpz_cpu PRIVATE /O2)
    else()
        target_compile_options(fofpz_cpu PRIVATE -O3 -march=native -g)
    endif()
endif()

# Enable warnings
if(MSVC)
    target_compile_options(fofpz_cpu PRIVATE /W4)
else()
    target_compile_options(fofpz_cpu PRIVATE -Wall -Wextra -pedantic -Wno-unused-parameter)
endif()

# MPI distributed target
find_package(MPI)
if(MPI_FOUND)
    add_executable(fofpz_cpu_mpi
        main_mpi.cpp
        fileIO.cpp
    )
    target_include_directories(fofpz_cpu_mpi PRIVATE ${ZSTD_INCLUDE_DIRS})
    target_link_libraries(fofpz_cpu_mpi PRIVATE ${ZSTD_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_compile_definitions(fofpz_cpu_mpi PRIVATE USE_MPI)
    if(USE_LIKWID)
        target_compile_definitions(fofpz_cpu_mpi PRIVATE USE_LIKWID)
        target_include_directories(fofpz_cpu_mpi PRIVATE ${LIKWID_INCLUDE_DIR})
        target_link_libraries(fofpz_cpu_mpi PRIVATE ${LIKWID_LIB})
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        if(MSVC)
            target_compile_options(fofpz_cpu_mpi PRIVATE /O2)
        else()
            target_compile_options(fofpz_cpu_mpi PRIVATE -O3 -march=native -g)
        endif()
    endif()
    if(MSVC)
        target_compile_options(fofpz_cpu_mpi PRIVATE /W4)
    else()
        target_compile_options(fofpz_cpu_mpi PRIVATE -Wall -Wextra -pedantic -Wno-unused-parameter)
    endif()
endif()
