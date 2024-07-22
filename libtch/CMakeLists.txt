
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(tch)

# Manually specify the Torch paths
set(Torch_DIR "/home/sreeraj/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

add_library(tch STATIC torch_api.cpp torch_api_generated.cpp)
target_link_libraries(tch "${TORCH_LIBRARIES}")
set_property(TARGET tch PROPERTY CXX_STANDARD 17)
target_compile_options(tch PRIVATE -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1)
target_link_libraries(tch c++)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})

install(TARGETS tch DESTINATION .)

