cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
project(collective_perception_cpp)

set(PROJECT_NAME_INTERFACE ${PROJECT_NAME}_interface)
set(PROTO_LIB ProtoLib) # define variable for ProtoLib

# Find dependencies
find_package(ARGoS REQUIRED)
find_package(Buzz REQUIRED)

# Add ARGoS and Buzz information
include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIR}) # include the */lib/argos3 directory to find the ARGoS libraries
link_libraries(${ARGOS_LDFLAGS})

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Buzz_DIR}")
include_directories(${BUZZ_C_INCLUDE_DIR})
include(UseBuzz)
set(BUZZ_BZZ_INCLUDE_DIR ${BUZZ_BZZ_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})

add_subdirectory(src) # add source directory
add_subdirectory(buzz) # add buzz directory
add_subdirectory(proto) # add proto directory

# Check if it's debug build
if(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_CONFIGURATION_TYPES MATCHES Debug)
    enable_testing()

    # Add Catch2 subdirectory for testing
    # see https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
    add_subdirectory(extern/Catch2)
    add_subdirectory(tests) # add test subdirectory
endif()