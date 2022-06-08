# Find dependencies
find_package(ARGoS REQUIRED)
find_package(Buzz REQUIRED)

include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIR}) # include the */lib/argos3 directory to find the ARGoS libraries
link_libraries(${ARGOS_LDFLAGS})

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Buzz_DIR}")
include_directories(${BUZZ_C_INCLUDE_DIR})

# Set install rule

# Build library
add_library(collective_perception_loop_functions
    MODULE
    collective_perception_loop_functions.cpp
    arena.cpp
    brain.cpp
)
target_link_libraries(collective_perception_loop_functions
    ${BUZZ_LIBRARY}
    argos3core_simulator
    argos3plugin_simulator_dynamics2d
    argos3plugin_simulator_entities
    argos3plugin_simulator_genericrobot
    argos3plugin_simulator_media
)
target_include_directories(collective_perception_loop_functions
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}> # include directory during build; which is "../include" from this directory
)