cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
project(collective_perception_cpp)

set(PROTO_LIB ProtoLib) # define variable for ProtoLib

add_subdirectory(src) # add source directory
add_subdirectory(buzz) # add buzz directory
add_subdirectory(proto) # add proto directory
