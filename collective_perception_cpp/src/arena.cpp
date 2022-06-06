#include "arena.hpp"
#include <iostream>
#include <assert.h>
#include <string>

Arena::Arena(const std::vector<uint32_t> &tile_count, const std::vector<float> &lower_lim_2d, const float &tile_size, const float &fill_ratio) :
    fill_ratio_(fill_ratio), tile_size_(tile_size)
{
    // Store arena dimensions
    num_tiles_ = Arena::Dimensions<uint32_t>(tile_count[0], tile_count[1]);
    lower_lim_ = Arena::Dimensions<float>(lower_lim_2d[0], lower_lim_2d[1]);

    // Create layout to the desired size with values of 0
    layout_.resize(num_tiles_.y, std::vector<uint32_t>(num_tiles_.x, 0) );
    GenerateTileArrangement();
}

void Arena::GenerateTileArrangement()
{
    std::default_random_engine generator;
    std::bernoulli_distribution distribution(fill_ratio_);

    // Populate the tiles with colors
    for (size_t i = 0; i < num_tiles_.y; ++i)
    {
        for (size_t j = 0; j < num_tiles_.x; ++j)
        {
            layout_[i][j] = static_cast<uint32_t>( distribution(generator) );
        }
    }
}

uint32_t Arena::GetColor(const float &x, const float &y)
{
    size_t tile_count_x = static_cast<size_t>( std::floor( (x - lower_lim_.x) / tile_size_) );
    size_t tile_count_y = static_cast<size_t>( std::floor( (y - lower_lim_.y) / tile_size_) );

    // Ensure that the requested counts are correct
    assert( tile_count_x < layout_[0].size() );
    assert( tile_count_y < layout_.size() );

    return layout_[tile_count_y][tile_count_x];
}