cmake_minimum_required(VERSION 3.18)
project(FOFHaloFinderParticleCompressor CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Create executable
add_executable(fofpz
    main.cu
    fileIO.cpp
    particle_compression.cu
    HuffmanZSTDCoder.cu
    statistics.cu
    FOFHaloFinder.cu
)

# Set CUDA properties
set_target_properties(fofpz PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89;90"
)

# Link libraries
target_link_libraries(fofpz PRIVATE 
    ${ZSTD_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
)

target_include_directories(fofpz PRIVATE 
    ${ZSTD_INCLUDE_DIRS}
)

# Enable optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(fofpz PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/O2>
        )
    else()
        target_compile_options(fofpz PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
    endif()
    
    # CUDA optimization flags
    target_compile_options(fofpz PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --ptxas-options=-v
        >
    )
endif()

# Enable warnings
if(MSVC)
    target_compile_options(fofpz PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(fofpz PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>
    )
endif()

# MPI distributed target
find_package(MPI)
if(MPI_FOUND)
    add_executable(fofpz_mpi
        main_mpi.cu
        fileIO.cpp
        particle_compression.cu
        HuffmanZSTDCoder.cu
        statistics.cu
        FOFHaloFinder.cu
    )

    set_target_properties(fofpz_mpi PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86;89;90"
    )

    target_link_libraries(fofpz_mpi PRIVATE
        ${ZSTD_LIBRARIES}
        CUDA::cudart
        CUDA::cuda_driver
        MPI::MPI_CXX
    )

    target_include_directories(fofpz_mpi PRIVATE
        ${ZSTD_INCLUDE_DIRS}
    )

    target_compile_definitions(fofpz_mpi PRIVATE USE_MPI)

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        if(MSVC)
            target_compile_options(fofpz_mpi PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/O2>
            )
        else()
            target_compile_options(fofpz_mpi PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
                $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            )
        endif()
        target_compile_options(fofpz_mpi PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --ptxas-options=-v
            >
        )
    endif()

    if(NOT MSVC)
        target_compile_options(fofpz_mpi PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>
        )
    endif()
endif()
