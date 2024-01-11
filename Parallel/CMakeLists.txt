cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_cmake)

# Enable CUDA language support
enable_language(CUDA)

# Add the executable compiled from the files
add_executable(cuda_app kernel.cu utils.cpp utils.h)

# Set the C++ standard to 11
set(CMAKE_CXX_STANDARD 11)

# Set the CUDA standard to 11
set(CMAKE_CUDA_STANDARD 11)

# Specify the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 61)

# Link the CUDA libraries
target_link_libraries(cuda_app cuda)