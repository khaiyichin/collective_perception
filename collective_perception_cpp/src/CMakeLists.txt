# Find dependencies
find_package(ARGoS REQUIRED)

# Set install rule

# Build library
add_library(collective_perception_loop_functions
    MODULE
    collective_perception_loop_functions.cpp
    arena.cpp
)
target_link_libraries(collective_perception_loop_functions
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