cmake_minimum_required(VERSION 3.16)
file(GLOB SRCS "./src/*.cpp")

find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

project(lacam)
add_library(${PROJECT_NAME} STATIC ${SRCS})
target_compile_options(${PROJECT_NAME} PUBLIC -O3 -Wall)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE ./include)

target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads "${TORCH_LIBRARIES}")

set_property(TARGET lacam PROPERTY POSITION_INDEPENDENT_CODE ON)