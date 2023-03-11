#ifndef BENCHMARK_ALGORITHM_HPP
#define BENCHMARK_ALGORITHM_HPP

#include <vector>
#include <string>
#include <random>
#include <functional>

// Buzz and ARGoS headers
#include <buzz/argos/buzz_loop_functions.h>
#include <buzz/buzzvm.h>
#include <argos3/core/utility/configuration/argos_configuration.h>
#include <nlohmann/json.hpp>

#include "util.hpp"
#include "data_common.hpp"

using namespace argos;
using namespace nlohmann;
// using (CBuzzLoopFunctions::*BuzzForeachVM)(std::function<void(const std::string &, buzzvm_t)>) = void;
using BuzzForeachVMFunc = std::function<void(CBuzzLoopFunctions::COperation &)>;

struct BuzzCOperationFunctorBase : public CBuzzLoopFunctions::COperation
{
    BuzzCOperationFunctorBase() {}

    BuzzCOperationFunctorBase(const std::string &id_prefix,
                              const int &id_base_num)
        : prefix(id_prefix),
          base_num(id_base_num) {}

    virtual ~BuzzCOperationFunctorBase() {}

    /**
     * @brief Strips the string prefix in the robot ID to obtain the numeric part
     *
     * @param str_robot_id Robot ID
     * @return int Numeric part of the robot ID
     */
    inline int GetNumericId(std::string str_robot_id)
    {
        return std::stoi(str_robot_id.erase(0, prefix.length()));
    }

    std::string prefix;

    int base_num;
};

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

    std::vector<json> json_data;
};

class BenchmarkAlgorithmBase
{
public:
    BenchmarkAlgorithmBase() {}

    BenchmarkAlgorithmBase(const BuzzForeachVMFunc &buzz_foreach_vm_func)
        // : buzz_foreach_vm_func_ptr_(std::make_shared<BuzzForeachVMFunc>(std::move(buzz_foreach_vm_func))) {}
        : buzz_foreach_vm_func_(buzz_foreach_vm_func)
    {
    }

    virtual ~BenchmarkAlgorithmBase() {}

    virtual void Init();

    virtual void PostStep();

    virtual void PostExperiment();

    virtual void SetupExperiment(const std::pair<double, double> &curr_paired_parameters);

    virtual void ComputeStats();

    virtual void InitializeJson(const std::pair<double, double> &curr_paired_parameters) {}

    virtual void WriteToJson() {}

    virtual BenchmarkDataBase &GetData();

    virtual std::vector<double> GetParameterRange();

protected:
    std::vector<json>::iterator curr_json_data_itr_;

    // std::shared_ptr<BuzzForeachVMFunc> buzz_foreach_vm_func_ptr_;
    BuzzForeachVMFunc buzz_foreach_vm_func_;
};

template <typename T>
class BenchmarkAlgorithmTemplate : public BenchmarkAlgorithmBase
{
public:
    BenchmarkAlgorithmTemplate() {}

    BenchmarkAlgorithmTemplate(const BuzzForeachVMFunc &buzz_foreach_vm_func)
        : BenchmarkAlgorithmBase(buzz_foreach_vm_func) {}

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