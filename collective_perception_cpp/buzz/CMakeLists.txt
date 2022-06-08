# Find dependencies
find_package(Buzz REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Buzz_DIR}")
include_directories(${BUZZ_C_INCLUDE_DIR})
include(UseBuzz)
set(BUZZ_BZZ_INCLUDE_DIR ${BUZZ_BZZ_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})

buzz_make(body.bzz)