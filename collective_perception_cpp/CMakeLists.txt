cmake_minimum_required(VERSION 3.15)
project(collective_perception_cpp)

include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIR})
link_libraries(${ARGOS_LDFLAGS})

add_subdirectory(src) # add source directory
# add_subdirectory(proto) # add proto directory