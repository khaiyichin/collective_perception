# See https://www.falkoaxmann.de/dev/2020/11/08/grpc-plugin-cmake-support.html
find_package(Protobuf REQUIRED)

set(PROTO_LIB_FILES
    ./util.proto
    ./simulation_data_set.proto
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # setting the fPIC flag

add_library(${PROTO_LIB} ${PROTO_LIB_FILES})
target_link_libraries(${PROTO_LIB} PUBLIC protobuf::libprotobuf)
target_include_directories(${PROTO_LIB}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

protobuf_generate(LANGUAGE cpp TARGET ${PROTO_LIB})
protobuf_generate(LANGUAGE python TARGET ${PROTO_LIB})