#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <string>
#include <ctime>

template <typename T>
std::vector<T> GenerateLinspace(const T &min, const T &max, const size_t &steps)
{
    // Compute increment
    T inc = (max - min) / static_cast<T>(steps - 1);

    // Populate vector
    std::vector<T> output(steps);
    T val;

    for (auto itr = output.begin(), val = min; itr != output.end(); ++itr, val += inc)
    {
        *itr = val;
    }

    return output;
}

inline std::string GetCurrentTimeStr()
{
    // Grab current local time
    time_t curr_time;
    time(&curr_time);
    tm *curr_tm = localtime(&curr_time);

    std::string datetime;
    datetime.resize(100);

    // Convert to string
    strftime(&(datetime[0]), datetime.size(), "%m%d%y_%H%M%S", curr_tm);
    return std::string(datetime.c_str());
}

inline std::string Round1000DoubleToStr(const double &val)
{
    return std::to_string(static_cast<int>(std::round(val * 1e3)));
}

#endif