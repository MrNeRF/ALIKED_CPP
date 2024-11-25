cmake_minimum_required(VERSION 3.26)
project(aliked LANGUAGES CUDA CXX)

# Determine the project root directory
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# Define the absolute path to libtorch
set(LIBTORCH_DIR "${PROJ_ROOT_DIR}/external/libtorch")

# Set CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_89,code=compute_89")

# --------------------------------------
# Core count calculation
# --------------------------------------

include(ProcessorCount)
ProcessorCount(total_cores)
if (NOT DEFINED total_cores OR total_cores EQUAL 0)
    set(total_cores 1)
    set(used_cores 1)
elseif (total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2") # use total_cores-2 if total_cores > 1
endif ()
set(ENV{MAKEFLAGS} "-j${used_cores}")
message(STATUS "Building with ${used_cores} out of ${total_cores} available cores")

# --------------------------------------
# Files
# --------------------------------------

set(HEADERS
        include/ALIKED.hpp
        include/blocks.hpp
        include/get_patches.hpp
        include/input_padder.hpp
        include/deform_conv2d.h
        include/get_patches_cuda.h
        include/cuda_helpers.h
)

set(SOURCES
        examples/main.cpp
        src/blocks.cpp
        src/ALIKED.cpp
        src/DKD.cpp
        src/get_patches.cpp
        src/SDDH.cpp
        src/deform_conv2d.cpp
        src/deform_conv2d_kernel.cu
        src/get_patches_cuda.cu
)

# --------------------------------------
# Project target
# --------------------------------------

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

set_source_files_properties(src/get_patches_cuda.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Torch setup
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Build options based on CMAKE_BUILD_TYPE
function(configure_build_type)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        #target_compile_options(${PROJECT_NAME} PRIVATE -O0 -g -G -Xcompiler -Werror)
        set_target_properties(${PROJECT_NAME} PROPERTIES
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                CUDA_SEPARABLE_COMPILATION ON
        )
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${PROJECT_NAME} PRIVATE -O3 -use_fast_math -Xcompiler -Ofast )
    endif ()
endfunction()
configure_build_type()

# Required packages
find_package(CUDAToolkit REQUIRED)

# Check CUDA version
if (CUDAToolkit_VERSION VERSION_LESS "12.1")
    message(FATAL_ERROR "This project requires CUDA 12.1 or higher")
endif ()

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# --------------------------------------
# Linking
# --------------------------------------

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
)