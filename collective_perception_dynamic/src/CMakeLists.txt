# Build library of data structures
add_library(collective_perception_data_structures
    SHARED
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

# Build dedicated test interface library @todo: hack to circumvent test issues
add_library(collective_perception_test_interface
    SHARED
    arena.cpp
    brain.cpp
)
target_link_libraries(collective_perception_test_interface
    PRIVATE
)
target_include_directories(collective_perception_test_interface
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}> # include directory during build; which is "../include" from this directory
)

# Build ARGoS loop function library
add_library(collective_perception_loop_functions
    SHARED
    collective_perception_loop_functions.cpp
    arena.cpp
    brain.cpp
    dac_plugin.cpp
)
target_link_libraries(collective_perception_loop_functions
    PUBLIC
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