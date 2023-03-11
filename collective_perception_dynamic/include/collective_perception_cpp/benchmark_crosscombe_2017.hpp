#ifndef BENCHMARK_CROSSCOMBE_2017_HPP
#define BENCHMARK_CROSSCOMBE_2017_HPP

#include <set>
#include <sstream>

#include "benchmark_algorithm.hpp"

// Define benchmark algorithm identifier; if modified then must change at other locations
#define CROSSCOMBE_2017 std::string("crosscombe_2017")

using RobotIdBeliefStrMap = std::unordered_map<std::string, std::vector<std::string>>;

/**
 * @brief Enumeration class for belief states
 *
 */
enum class BeliefState
{
    negative = 0,
    indeterminate,
    positive
};

enum class RobotState
{
    signalling = 0,
    updating
};

/**
 * @brief Functor to process the robot belief
 *
 */
struct ProcessRobotBelief : BuzzCOperationFunctorBase
{
    ProcessRobotBelief() {}

    ProcessRobotBelief(const std::string &id_prefix,
                       const int &id_base_num,
                       const int &num_options,
                       const int &num_robots,
                       const std::vector<int> &malf_robot_ids,
                       const std::vector<unsigned int> &option_qualities,
                       const std::shared_ptr<RobotIdBeliefStrMap> &id_belief_map_ptr)
        : BuzzCOperationFunctorBase(id_prefix, id_base_num),
          num_options(num_options),
          num_robots(num_robots),
          malf_robot_ids(malf_robot_ids),
          option_qualities(option_qualities),
          id_belief_map_ptr(id_belief_map_ptr)
    {
        // Initialize generator
        std::random_device rd;
        generator = std::mt19937(rd());
    }

    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    inline std::string ConvertBeliefVecToString(const std::vector<int> &belief_vec)
    {
        std::stringstream ss;

        std::copy(belief_vec.begin(), belief_vec.end(), std::ostream_iterator<int>(ss, ""));

        return ss.str();
    }

    inline void PopulateSelfBeliefVecInVM(buzzvm_t t_vm, const std::vector<int> self_belief)
    {
        BuzzTableOpen(t_vm, "self_belief");

        for (int i = 0; i < num_options; ++i)
        {
            BuzzTablePut(t_vm, std::to_string(i), self_belief[i]);
        }

        BuzzTableClose(t_vm);
    }

    inline std::vector<int> ExtractSelfBeliefVecFromVM(buzzvm_t t_vm)
    {
        std::vector<int> belief_vec;
        belief_vec.reserve(num_options);

        BuzzTableOpen(t_vm, "self_belief");

        for (int i = 0; i < num_options; ++i)
        {
            belief_vec.push_back(buzzobj_getint(BuzzTableGet(t_vm, std::to_string(i))));
        }

        BuzzTableClose(t_vm);

        return belief_vec;
    }

    /**
     * @brief Update belief based on a signalling neighbor's belief
     *
     * @param self_belief_vec Beliefs of the current robot
     * @param signalled_belief_vec Beliefs of the selected neighboring robot
     * @return std::vector<int> Updated beliefs of the current robot
     */
    std::vector<int> UpdateSelfBeliefVec(const std::vector<int> &self_belief_vec,
                                         const std::vector<int> &signalled_belief_vec);

    inline std::vector<int> GenerateBeliefVecFromRandomOption(const std::vector<int> options_indices)
    {
        std::vector<int> selection(1);

        std::sample(options_indices.begin(),
                    options_indices.end(),
                    selection.begin(),
                    1,
                    generator);

        std::vector<int> belief(num_options, 0);
        belief[*selection.begin()] = static_cast<int>(BeliefState::positive);

        return belief;
    }

    /**
     * @brief Get the truth value for a single option
     *
     * @param self_belief Belief of the current robot
     * @param signalled_belief Belief of the signaling robot
     * @return int Updated truth value for the current robot's belief
     */
    int GetTruthValue(const int &self_belief,
                      const int &signalled_belief);

    /**
     * @brief Get the broadcast duration
     *
     * @param option_id Index of the option
     * @return int Duration in units of ticks
     */
    inline int GetBroadcastDuration(const unsigned int &option_id) { return option_qualities[option_id]; }

    inline bool IsMalfunctioning(const std::string &str_robot_id)
    {
        return std::find(malf_robot_ids.begin(), malf_robot_ids.end(), GetNumericId(str_robot_id)) != malf_robot_ids.end();
    }

    std::mt19937 generator;

    int num_options;

    int num_robots;

    bool initialized = false;

    std::set<std::string> initialized_robot_ids;

    std::vector<int> malf_robot_ids;

    std::vector<unsigned int> option_qualities;

    std::shared_ptr<RobotIdBeliefStrMap> id_belief_map_ptr;
};

struct BenchmarkDataCrosscombe2017 : BenchmarkDataBase
{
    BenchmarkDataCrosscombe2017() : BenchmarkDataBase("benchmark_" + CROSSCOMBE_2017, "flawed_robot_ratio") {}

    std::vector<double> flawed_ratio_range;

    unsigned int num_possible_options;

    unsigned int total_quality_score = 100;
};

class BenchmarkCrosscombe2017 : public BenchmarkAlgorithmTemplate<BenchmarkDataCrosscombe2017>
{
public:
    BenchmarkCrosscombe2017(const BuzzForeachVMFunc &buzz_foreach_vm_func, TConfigurationNode &t_tree);

    void SetupExperiment(const std::pair<double, double> &curr_paired_parameters);

    void PostStep();

    void PostExperiment();

    void InitializeJson(const std::pair<double, double> &curr_paired_parameters);

    void ComputeStats();

    void WriteToJson();

    std::vector<double> GetParameterRange() { return data_.flawed_ratio_range; }

    // std::string GetParameterString() { return data_.parameter_keyword; }

    void ComputeOptionQualities(const double &curr_tfr);

private:
    ProcessRobotBelief process_robot_belief_functor_;

    std::vector<unsigned int> curr_option_qualities_;

    std::shared_ptr<RobotIdBeliefStrMap> id_belief_map_ptr_;
};

#endif