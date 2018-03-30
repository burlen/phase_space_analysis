cmake_minimum_required(VERSION 2.8)

PROJECT(density)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(density MACOSX_BUNDLE density.cpp)
target_link_libraries(density ${VTK_LIBRARIES})
