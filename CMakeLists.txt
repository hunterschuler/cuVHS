cmake_minimum_required(VERSION 3.25)
project(cuVHS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for all modern GPU architectures unless user specifies
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
endif()

find_package(CUDAToolkit REQUIRED)

# cuFFT is required for FM demodulation and chroma filtering
find_library(CUFFT_LIBRARY cufft HINTS ${CUDAToolkit_LIBRARY_DIR})

add_executable(cuvhs
    src/main.cpp
    src/gpu/device.cu
    src/io/raw_reader.cpp
    src/io/tbc_writer.cpp
    src/format/video_format.cpp
    src/pipeline/pipeline.cu
    src/pipeline/fm_demod.cu
    src/pipeline/sync_pulses.cu
    src/pipeline/line_locs.cu
    src/pipeline/hsync_refine.cu
    src/pipeline/lineloc_coherence.cu
    src/pipeline/tbc_resample.cu
    src/pipeline/chroma_decode.cu
    src/pipeline/dropout_detect.cu
)

target_include_directories(cuvhs PRIVATE src)
target_link_libraries(cuvhs PRIVATE
    CUDA::cudart
    CUDA::cufft
)
