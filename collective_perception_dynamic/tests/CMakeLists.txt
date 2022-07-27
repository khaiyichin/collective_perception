# Define unit tests to build
set(UNIT_TEST_SOURCES
    test_header.hpp
    test_brain.cpp
    test_arena.cpp
)

set(TEST_TARGET_NAME tests) # define test executable name

# Built test
add_executable(${TEST_TARGET_NAME}
    ${UNIT_TEST_SOURCES}
)

target_link_libraries(${TEST_TARGET_NAME}
    PRIVATE
    collective_perception_test_interface
    ${PROTO_LIB}
    Catch2::Catch2WithMain
)

target_include_directories(${TEST_TARGET_NAME}
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}> # include directory during build; which is "../include" from this directory
)

add_test(
    NAME ${TEST_TARGET_NAME}
    COMMAND ${TEST_TARGET_NAME} -r xml -o test_report.xml
)