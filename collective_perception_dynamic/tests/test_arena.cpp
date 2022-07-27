#include <utility>

#include "test_header.hpp"
#include "arena.hpp"

#include <iostream> // for debugging

TEST_CASE("Arena class", "[ArenaCls]")
{
    // Initialize random number generators
    std::random_device rd;
    std::default_random_engine generator = std::default_random_engine(rd());
    std::uniform_int_distribution<int> tile_count_distribution(TILE_COUNT_MIN, TILE_COUNT_MAX);
    std::uniform_real_distribution<float> tile_size_distribution(TILE_SIZE_MIN, TILE_SIZE_MAX);
    std::uniform_real_distribution<float> target_fill_ratio_distribution(0, 1.0);

    SECTION("test tile distribution", "[method]")
    {
        std::vector<float> true_tfr;
        std::vector<float> computed_tfr;

        std::vector<int> true_tiles;
        std::vector<int> computed_tiles;

        for (size_t i = 0; i < TEST_REPEATS; ++i)
        {
            /* The assumption here is that the arena and tiles are square. */
            int tile_count = tile_count_distribution(generator);
            float tfr = target_fill_ratio_distribution(generator);
            float tile_size = tile_size_distribution(generator);

            true_tfr.push_back(tfr);
            true_tiles.push_back(tile_count * tile_count);

            // Compute arena limits (ordinarily, this information is provided by the ARGoS simulator)
            float arena_size = tile_size * tile_count;
            float lower_limit_x = -arena_size / 2;
            float lower_limit_y = lower_limit_x;

            std::pair<unsigned int, unsigned int> tile_count_2d(tile_count, tile_count);
            std::pair<float, float> lower_limit_2d(lower_limit_x, lower_limit_y);

            Arena a(tile_count_2d, lower_limit_2d, tile_size, tfr);

            // Generate some tiles using distribution parameter
            a.GenerateTileArrangement();

            // Store tile and fill ratio information
            computed_tiles.push_back(a.GetTotalNumTiles());
            computed_tfr.push_back(a.GetTrueTileDistribution());
        }

        // Verify that the computed target fill ratio matches the intended fill ratio
        REQUIRE_THAT(computed_tfr, Catch::Matchers::Approx(true_tfr).margin(1e-2));

        // Verify the total number of tiles
        REQUIRE(computed_tiles == true_tiles);
    }
}