#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <limits>
#include <random>

#define TEST_REPEATS 50 // number of repeats for a test
#define OBSERVATION_MAX 1e6 // upper bound for a random number of observations; prevent rounding errors during computation
#define NEIGHBOR_AGENTS_MAX 1e3 // upper bound for a random number of agent neighbors
#define CONFIDENCE_MAX 1e3 // upper bound on random confidence value (useful for preventing skewed averages)