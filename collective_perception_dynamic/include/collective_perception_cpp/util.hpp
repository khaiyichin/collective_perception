#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <string>
#include <ctime>
#include <random> // mt19937, uniform_real_distribution, normal_distribution
#include <algorithm> // sample
#include <stdexcept>

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

/**
 * @brief Draw a sample of robot IDs without replacement
 *
 * @param num_robots_to_sample Number of robots to sample
 * @param robot_id_vec Vector of robot IDs to sample from
 * @return std::vector<std::string> Drawn robot IDs
 */
inline std::vector<std::string> SampleRobotIdsWithoutReplacement(const unsigned int &num_robots_to_sample,
                                                                 const std::vector<std::string> &robot_id_vec)
{
    // Sample random robot IDs (without replacement)
    std::vector<std::string> sampled_robot_ids;

    std::sample(robot_id_vec.begin(),
                robot_id_vec.end(),
                std::back_inserter(sampled_robot_ids),
                num_robots_to_sample,
                std::mt19937{std::random_device{}()});

    return sampled_robot_ids;
}

/**
 * @brief  Encode robot sensor probabilities
 *
 * The output is used to generate robots with heterogenous sensor probabilities
 *
 * Since the distribution is for the sensor probability, the possible range is [0.0, 1.0].
 * A 3 decimal point precision is enforced, so the encoding will scale the values up by 1e3.
 *
 * Encoding output rules:
 *      The first element is the negative sign.
 *      The second element is either 2 (uniform distribution) or 3 (normal distribution).
 *      The 3rd - 6th element indicates the first distribution parameter, scaled by 1e3.
 *      The 7th - 10th element indicates the second distribution parameter, scaled by 1e3.
 *
 *  Example:
 *      EncodeSensorDistParams(-2, 0.525, 0.975); // this gives -205250975, which means a uniform distribution
 *                                                // with lower bound 0.525, upper bound 0.975
 *
 *      EncodeSensorDistParams(-3, 1.0, 0.4); // this gives -310000400, which means a normal distribution
 *
 * @param id Distribution identifier: -2 = uniform, -3 = normal
 * @param param_1 Uniform: lower bound; Normal: mean
 * @param param_2 Uniform: upper bound; Normal: variance
 * @return double Encoded value
 */
inline double EncodeSensorDistParams(const int &id, const double &param_1, const double &param_2)
{
    // Define lambda function to scale and convert values to string
    auto scale_and_convert_to_str = [](const double &val)
    {
        // Scale values and convert to string
        std::string scaled_val_str = std::to_string(int(std::round(val * 1e3)));

        // Append leading zeros
        if (scaled_val_str.length() < 4)
        {
            scaled_val_str.insert(0, 4 - scaled_val_str.length(), '0');
        }

        return scaled_val_str;
    };

    return std::stod(std::to_string(id) + scale_and_convert_to_str(param_1) + scale_and_convert_to_str(param_2));
}

/**
 * @brief Generate a sensor probability at random based on a probability distribution
 *
 * @param encoded_prob_dist Encoded value (see EncodeSensorDistParams)
 * @param generator mt19937 random number generator
 * @return float Sampled random sensor probability
 */
inline float GenerateRandomSensorProbability(const double &encoded_prob_dist,
                                             std::mt19937 &generator)
{
    // Decode sensor probability
    std::string encoded_str = std::to_string(encoded_prob_dist);

    // Grab the distribution identifier
    int id = std::stoi(encoded_str.substr(0, 2));

    // Grab the two parameters
    auto decode_sp = [](const std::string &str)
    {
        return std::stof(str) / 1e3;
    };

    float param_1(decode_sp(encoded_str.substr(2, 4)));
    float param_2(decode_sp(encoded_str.substr(6, 4)));

    switch (id)
    {
    case -2: // uniform distribution
    {
        std::uniform_real_distribution<float> dist(param_1, param_2);
        return dist(generator);
    }

    case -3: // normal distribution
    {
        std::normal_distribution<float> dist(param_1, std::sqrt(param_2));
        return dist(generator);
    }

    default:
        throw std::invalid_argument("Encoded string does not match expected values.");
        return 0.0;
    }
}

#endif