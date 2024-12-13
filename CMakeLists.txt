cmake_minimum_required(VERSION 3.30)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)
project(CUCKOO_HASH LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(tester SHARED tester/test_case.cu)
add_library(opt_tester SHARED tester/opt_test_case.cu)

add_executable(main main.cu)
target_link_libraries(main tester)
target_link_libraries(main opt_tester)

add_executable(task1 task1.cu)
target_link_libraries(task1 tester)

add_executable(task2 task2.cu)
target_link_libraries(task2 tester)

add_executable(task3 task3.cu)
target_link_libraries(task3 opt_tester)

add_executable(task4 task4.cu)
target_link_libraries(task4 opt_tester)