cmake_minimum_required(VERSION 3.10)
project(sr_bench CXX)

set(CMAKE_CXX_STANDARD 14)

if(NOT DEFINED NCNN_DIR)
    message(FATAL_ERROR "Pass -DNCNN_DIR=/path/to/ncnn/build-android-aarch64/install")
endif()

add_executable(sr_bench main.cpp)
target_include_directories(sr_bench PRIVATE ${NCNN_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(sr_bench PRIVATE ${NCNN_DIR}/lib)
target_link_libraries(sr_bench ncnn)

if(ANDROID)
    target_link_libraries(sr_bench log android)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(sr_bench OpenMP::OpenMP_CXX)
    endif()
endif()
