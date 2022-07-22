# Find dependencies
find_package(ARGoS REQUIRED)
find_package(Buzz REQUIRED)

include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIR}) # include the */lib/argos3 directory to find the ARGoS libraries
link_libraries(${ARGOS_LDFLAGS})

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Buzz_DIR}")
include_directories(${BUZZ_C_INCLUDE_DIR})

# Set install rule

# Build libraries
add_library(collective_perception_data_structures
simulation_stats_set.cpp
simulation_agent_data_set.cpp
)
target_link_libraries(collective_perception_data_structures
PRIVATE
${PROTO_LIB}
)
target_include_directories(collective_perception_data_structures
PRIVATE
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}> # include directory during build; which is "../include" from this directory
)

add_library(collective_perception_loop_functions
    MODULE
    collective_perception_loop_functions.cpp
    arena.cpp
    brain.cpp
)
target_link_libraries(collective_perception_loop_functions
    ${BUZZ_LIBRARY}
    ${PROTO_LIB}
    collective_perception_data_structures
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

# Build executables
add_executable(run_dynamic_simulations run_dynamic_simulations.cpp)
target_link_libraries(run_dynamic_simulations
    PRIVATE
    argos3core_simulator
    ${PROTO_LIB}
)

add_executable(protobuf_info protobuf_info.cpp)
target_link_libraries(protobuf_info
    PRIVATE
    collective_perception_data_structures
    ${PROTO_LIB}
)
target_include_directories(protobuf_info
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}> # include directory during build; which is "../include" from this directory
)

add_executable(compute_wall_positions compute_wall_positions.cpp)