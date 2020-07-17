cmake_minimum_required(VERSION 3.16)
project(peer_access_without_enabling_it CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(peer_access_without_enabling_it main.cu)
target_link_libraries(peer_access_without_enabling_it PRIVATE cuda)