cmake_minimum_required(VERSION 3.25)
project(aliked_cpp LANGUAGES CUDA CXX)

# Set CUDA and LibTorch paths
set(LIBTORCH_DIR "external/libtorch") # Replace with the correct path
set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

# Include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

# Define source files
set(ALIKED_SOURCES
        src/ALIKED.cpp
        src/DKD.cpp
        src/SDDH.cpp
)

set(CUSTOM_OPS_SOURCES
        src/get_patches.cpp
        src/get_patches_cuda.cu
)

# Custom CUDA operations
add_library(custom_ops SHARED ${CUSTOM_OPS_SOURCES})

target_link_libraries(custom_ops
        ${TORCH_LIBRARIES}
        ${CUDA_LIBRARIES}
)

set_target_properties(custom_ops PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CXX_VISIBILITY_PRESET "hidden"
)

# Main ALIKED library
add_library(aliked SHARED ${ALIKED_SOURCES})

target_include_directories(aliked PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aliked PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        custom_ops
)

# Example executable
add_executable(aliked_example examples/main.cpp)
target_link_libraries(aliked_example
        aliked
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)