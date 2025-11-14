cmake_minimum_required(VERSION 3.16)
project(lacam-project CXX)

add_subdirectory(./lacam)
add_subdirectory(./third_party/argparse)

add_executable(main main.cpp)
target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main lacam argparse)

add_library(lagat SHARED main.cpp)
target_compile_features(lagat PUBLIC cxx_std_17)
target_include_directories(lagat PRIVATE ./lacam/include ./third_party/argparse)
target_link_libraries(lagat lacam libpthread.so argparse)
