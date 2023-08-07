# Find dependencies
find_package(Buzz REQUIRED)

buzz_make(body.bzz INCLUDES body_common.bzz)
buzz_make(body_crosscombe_2017.bzz INCLUDES body_common.bzz)
buzz_make(body_ebert_2020.bzz INCLUDES body_common.bzz)