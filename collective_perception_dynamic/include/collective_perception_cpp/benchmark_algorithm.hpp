#ifndef BENCHMARK_ALGORITHM_HPP
#define BENCHMARK_ALGORITHM_HPP

#include <vector>
#include <string>
#include <random>

// Buzz and ARGoS headers
#include <buzz/argos/buzz_loop_functions.h>
#include <buzz/buzzvm.h>
#include <argos3/core/utility/configuration/argos_configuration.h>
#include <nlohmann/json.hpp>

#include "util.hpp"
#include "data_common.hpp"

using namespace nlohmann;

struct BenchmarkDataBase
{
    BenchmarkDataBase(const std::string &sim_type, const std::string &param_str)
        : simulation_type(sim_type), parameter_keyword(param_str) {}

    std::string simulation_type;

    std::string parameter_keyword;

    unsigned int num_agents;

    unsigned int num_trials;

    unsigned int num_steps;

    int id_base_num;

    float comms_range;

    float density;

    float speed;

    std::string id_prefix;

    std::string output_filepath;

    std::vector<double> tfr_range;

    std::vector<json> data;
};

class BenchmarkAlgorithmBase
{
public:
    virtual ~BenchmarkAlgorithmBase() {}

    virtual void Init();

    virtual void PostStep();

    virtual void PostExperiment();

    virtual void SetupExperiment(const std::pair<double, double> &curr_paired_parameters);

    virtual void InitializeJson(const std::pair<double, double> &curr_paired_parameters) {}

    virtual void WriteToJson() {}

    virtual BenchmarkDataBase &GetData();

    virtual std::vector<double> GetParameterRange();

protected:
    json curr_json_;
};

template <typename T>
class BenchmarkAlgorithmTemplate : public BenchmarkAlgorithmBase
{
public:
    BenchmarkAlgorithmTemplate() {}

    std::string GetParameterString() { return data_.parameter_str; }

    T &GetData() { return data_; }

protected:
    std::vector<int> SampleRobotIdsWithoutReplacement(const unsigned int &num_robots_to_sample, const unsigned int &starting_base_num)
    {
        // Create a vector of all IDs
        std::vector<int> ids(data_.num_agents);
        std::iota(ids.begin(), ids.end(), starting_base_num);

        // Sample random robot IDs (without replacement)
        std::vector<int> sampled_robot_ids;
        std::sample(ids.begin(), ids.end(), std::back_inserter(sampled_robot_ids), num_robots_to_sample, std::mt19937{std::random_device{}()});

        return sampled_robot_ids;
    }

    T data_;
};

#endif