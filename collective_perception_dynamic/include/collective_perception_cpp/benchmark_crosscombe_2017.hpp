#ifndef BENCHMARK_CROSSCOMBE_2017_HPP
#define BENCHMARK_CROSSCOMBE_2017_HPP

#include <set>
#include <sstream>

#include "benchmark_algorithm.hpp"

// Define default benchmark algorithm identifiers; if modified then must change at other locations
#define CROSSCOMBE_2017 std::string("crosscombe_2017")
#define CROSSCOMBE_2017_PARAM std::string("flawed_robot_ratio")
#define CROSSCOMBE_2017_PARAM_ABBR std::string("frr")

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

/**
 * @brief Enumeration class for robot states
 *
 */
enum class RobotState
{
    signalling = 0,
    updating
};

/**
 * @brief Functor to process the robots' belief
 *
 */
struct ProcessRobotBelief : public CBuzzLoopFunctions::COperation
{
    /**
     * @brief Construct a new ProcessRobotBelief object
     *
     */
    ProcessRobotBelief() {}

    /**
     * @brief Construct a new ProcessRobotBelief object
     *
     * @param num_options Number of options available for the robots to select from
     * @param num_robots Number of robots to simulate
     * @param speed Speed of the robot in cm/s
     * @param flawed_robot_ids IDs of flawed robots
     * @param option_qualities Qualities of options
     * @param id_belief_map_ptr Pointer to the map containing the robot IDs and their beliefs
     */
    inline ProcessRobotBelief(const int &num_options,
                              const int &num_robots,
                              const float &speed,
                              const std::vector<std::string> &flawed_robot_ids,
                              const std::vector<unsigned int> &option_qualities,
                              const std::shared_ptr<RobotIdBeliefStrMap> &id_belief_map_ptr)
        : num_options(num_options),
          num_robots(num_robots),
          spd(speed),
          flawed_robot_ids(flawed_robot_ids),
          option_qualities(option_qualities),
          id_belief_map_ptr(id_belief_map_ptr)
    {
        // Initialize generator
        std::random_device rd;
        generator = std::mt19937(rd());

        // Initialize random belief distribution (for flawed robots)
        flawed_belief_dist = std::uniform_int_distribution(static_cast<int>(BeliefState::negative),
                                                           static_cast<int>(BeliefState::positive));
    }

    /**
     * @brief Overload the () operator
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    /**
     * @brief Convert the belief (a vector of ints) into a string
     *
     * @param belief_vec Belief vector
     * @return std::string Converted belief
     */
    inline std::string ConvertBeliefVecToString(const std::vector<int> &belief_vec)
    {
        std::stringstream ss;

        std::copy(belief_vec.begin(), belief_vec.end(), std::ostream_iterator<int>(ss, ""));

        return ss.str();
    }

    /**
     * @brief Populate the self belief of the robot in Buzz
     *
     * @param t_vm Buzz VM
     * @param self_belief_vec Belief to populate into Buzz
     */
    inline void PopulateSelfBeliefVecInVM(buzzvm_t t_vm, const std::vector<int> &self_belief_vec)
    {
        BuzzTableOpen(t_vm, "self_belief");

        for (int i = 0; i < num_options; ++i)
        {
            BuzzTablePut(t_vm, i, self_belief_vec[i]);
        }

        BuzzTableClose(t_vm);
    }

    /**
     * @brief Populate the broadcast duration in Buzz
     *
     * @param t_vm Buzz VM
     * @param duration Duration in ticks
     */
    inline void PopulateBroadcastDurationInVM(buzzvm_t t_vm, const int &duration)
    {
        BuzzPut(t_vm, "broadcast_duration", duration);
    }

    /**
     * @brief Extract the self belief of the robot from Buzz
     *
     * @param t_vm Buzz VM
     * @return std::vector<int> Robot's self belief
     */
    inline std::vector<int> ExtractSelfBeliefVecFromVM(buzzvm_t t_vm)
    {
        std::vector<int> belief_vec;
        belief_vec.reserve(num_options);

        BuzzTableOpen(t_vm, "self_belief");

        for (int i = 0; i < num_options; ++i)
        {
            belief_vec.push_back(buzzobj_getint(BuzzTableGet(t_vm, i)));
        }

        BuzzTableClose(t_vm);

        return belief_vec;
    }

    /**
     * @brief Update the self belief vector
     *
     * If `randomize_belief` is true, the second member of `self_and_signalled_beliefs` can be empty;
     * this method calls the `NormalizeAndPopulateSelfBeliefVec` so the self belief vector and broadcast
     * duration will be populated.
     *
     * @param t_vm Buzz VM
     * @param self_and_signalled_beliefs Pair of self belief vector and chosen signalled belief vector
     * @param randomize_belief Flag to randomize updated self belief vector
     * @return std::vector<int> Updated self belief
     */
    std::vector<int> UpdateSelfBeliefVec(buzzvm_t t_vm,
                                         const std::pair<std::vector<int>, std::vector<int>> &self_and_signalled_beliefs,
                                         const bool &randomize_belief = false);

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

    /**
     * @brief Normalize the self belief vector and populate into the VM
     *
     * @param t_vm Buzz VM
     * @param self_belief_vec Self belief vector
     * @return std::vector<int> Normalized self belief vector
     */
    std::vector<int> NormalizeAndPopulateSelfBeliefVec(buzzvm_t t_vm, const std::vector<int> &self_belief_vec);

    /**
     * @brief Get the index of the positive belief in a self belief vector
     *
     * @param self_belief_vec Self belief vector of a robot
     * @return int Index of the positive belief
     */
    inline int GetPositiveBeliefIndex(const std::vector<int> &self_belief_vec)
    {
        return std::find(self_belief_vec.begin(),
                         self_belief_vec.end(),
                         static_cast<int>(BeliefState::positive)) -
               self_belief_vec.begin();
    }

    /**
     * @brief Get the indices of indeterminate beliefs in a self belief vector
     *
     * @param self_belief_vec Self belief vector of a robot
     * @return std::vector<int> Indices of the indeterminate beliefs
     */
    std::vector<int> GetIndeterminateBeliefIndices(const std::vector<int> &self_belief_vec);

    /**
     * @brief Check if the robot is flawed
     *
     * @param str_robot_id Robot ID
     * @return true
     * @return false
     */
    inline bool IsFlawed(const std::string &str_robot_id)
    {
        return std::find(flawed_robot_ids.begin(),
                         flawed_robot_ids.end(),
                         str_robot_id) != flawed_robot_ids.end();
    }

    std::mt19937 generator; ///< Random number generator

    int num_options; ///< Total number of available options to choose from

    int num_robots; ///< Number of robots to simulate

    float spd; ///< Maximum linear robot speed in cm/s

    bool initialized = false; ///< Flag to indicate if robot initialization has occurred

    std::uniform_int_distribution<int> flawed_belief_dist; ///< Uniform distribution for generating beliefs for flawed robots

    std::set<std::string> initialized_robot_ids; ///< Set of robot IDs that have been initialized

    std::vector<std::string> flawed_robot_ids; ///< IDs of flawed robots

    std::vector<unsigned int> option_qualities; ///< Qualities of options

    std::shared_ptr<RobotIdBeliefStrMap> id_belief_map_ptr; ///< Pointer to the map containing the robot IDs and their beliefs
};

/**
 * @brief Struct for the Crosscombe 2017 benchmark data
 *
 */
struct BenchmarkDataCrosscombe2017 : BenchmarkDataBase
{
    /**
     * @brief Construct a new BenchmarkDataCrosscombe2017 object
     *
     */
    BenchmarkDataCrosscombe2017() : BenchmarkDataBase(CROSSCOMBE_2017) {}

    std::vector<double> &flawed_ratio_range = benchmark_param_range; ///< Range of flawed robot ratios

    unsigned int num_possible_options; ///< Number of possible options to choose from

    unsigned int total_quality_score = 100; ///< Total quality score to normalize the individual qualities
};

/**
 * @brief Class to implement the Crosscombe 2017 algorithm
 *
 */
class BenchmarkCrosscombe2017 : public BenchmarkAlgorithmTemplate<BenchmarkDataCrosscombe2017>
{
public:
    /**
     * @brief Construct a new BenchmarkCrosscombe2017 object
     *
     * @param buzz_foreach_vm_func COperation::BuzzForeachVM functor
     * @param t_tree XML node tree with `algorithm` as the root node
     * @param robot_id_vec Vector of all robot IDs
     */
    BenchmarkCrosscombe2017(const BuzzForeachVMFunc &buzz_foreach_vm_func,
                            TConfigurationNode &t_tree,
                            const std::vector<std::string> &robot_id_vec);

    /**
     * @brief Initialize the algorithm
     *
     */
    inline void Init() {}

    /**
     * @brief Setup the experiment
     *
     * @param trial_ind Current trial index
     * @param curr_paired_parameters Current paired parameters
     */
    void SetupExperiment(const int &trial_ind, const std::pair<double, double> &curr_paired_parameters);

    /**
     * @brief Execute post Step() (provided in argos::CSimulator) operations
     *
     */
    void PostStep();

    /**
     * @brief Execute post experiment operations
     *
     * @param final_experiment Flag to indicate whether this has been the last experiment
     */
    void PostExperiment(const bool &final_experiment);

    /**
     * @brief Initialize the JSON object
     *
     */
    void InitializeJson();

    /**
     * @brief Save the JSON data
     *
     * @param foldername_prefix Prefix for the high-level folder
     */
    void SaveData(const std::string &foldername_prefix = "");

    /**
     * @brief Get the range of the benchmark-specific parameter
     *
     * @return std::vector<double> Range of the parameter
     */
    inline std::vector<double> GetParameterRange() { return data_.flawed_ratio_range; }

    /**
     * @brief Get the benchmark-specific parameter keyword
     *
     * @return std::string Parameter keyword
     */
    inline std::string GetParameterKeyword() { return CROSSCOMBE_2017_PARAM; }

    /**
     * @brief Get the abbreviated benchmark-specific parameter keyword
     *
     * @return std::string Abbreviated parameter keyword
     */
    inline std::string GetParameterKeywordAbbr() { return CROSSCOMBE_2017_PARAM_ABBR; }

private:
    /**
     * @brief Compute the option qualities
     *
     * @param curr_tfr Current target fill ratio
     */
    void ComputeOptionQualities(const double &curr_tfr);

    int curr_num_flawed_robots_; ///< Current number of flawed robots

    ProcessRobotBelief process_robot_belief_functor_; ///< Functor to process robot beliefs

    std::vector<unsigned int> curr_option_qualities_; ///< Current option qualities

    std::shared_ptr<RobotIdBeliefStrMap> id_belief_map_ptr_; ///< Pointer to the map containing the robot IDs and their beliefs
};

#endif