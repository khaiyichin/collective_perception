#ifndef ARENA_HPP
#define ARENA_HPP

#include <vector>
#include <random>
#include <cmath>
#include <assert.h>

class Arena
{
    template <typename T>
    struct Dimensions
    {
        T x;
        T y;
        Dimensions<T>() {}
        Dimensions<T>(const T &x, const T &y) : x(x), y(y) {}
    };

public:

    Arena() {};
    Arena(const std::vector<uint32_t> &tile_count, const std::vector<float> &lower_lim_2d, const float &tile_size, const float &fill_ratio);
    ~Arena() {};

    /**
     * @brief Generate the tiles.
     * 
     */
    void GenerateTileArrangement();
    uint32_t GetColor(const float &x, const float &y);

private:
    float fill_ratio_;
    float tile_size_;
    Dimensions<uint32_t> num_tiles_;
    Dimensions<float> lower_lim_;

    std::vector<std::vector<uint32_t>> layout_;
};

#endif