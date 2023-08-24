#ifndef BENCHMARK_VALENTINI_2016_HPP
#define BENCHMARK_VALENTINI_2016_HPP

#include "benchmark_algorithm.hpp"

// Define default benchmark algorithm identifiers; if modified then must change at other locations
#define VALENTINI_2016 std::string("valentini_2016")
#define VALENTINI_2016_PARAM std::string("sensor_probability")
#define VALENTINI_2016_PARAM_ABBR std::string("sp")

using RobotIdDataStrMap = std::unordered_map<std::string, std::vector<std::string>>; // at each time step, the data string would be "<STATE>,<WHITE-QUALITY>,<BLACK-QUALITY>,<OPINION>"

struct ProcessRobotOpinions : public CBuzzLoopFunctions::COperation
{
    enum class RobotState
    {
        exploration = 0,
        dissemination = 1
    };

    enum class RobotOpinion
    {
        white = 0,
        black = 1
    };

    struct RobotDataStruct
    {
        int current_state;

        int current_opinion;

        std::array<double, 2> opinion_qualities{0.0, 0.0};

        int current_exp_duration;

        int current_dis_duration;

        int duration_tracker;
    };

    ProcessRobotOpinions() {}

    inline ProcessRobotOpinions(const int &num_robots,
                                const float &speed,
                                const double &b_prob,
                                const double &w_prob,
                                const double &exp_mean_dur_ticks,
                                const double &dis_mean_dur_fac_ticks,
                                const bool &voter_model,
                                const std::shared_ptr<RobotIdDataStrMap> &id_data_str_map_ptr)
        : num_robots(num_robots),
          spd(speed),
          b_prob(b_prob),
          w_prob(w_prob),
          exp_mean_dur_ticks(exp_mean_dur_ticks),
          dis_mean_dur_fac_ticks(dis_mean_dur_fac_ticks),
          voter_model(voter_model),
          id_data_str_map_ptr(id_data_str_map_ptr)
    {
        // Initialize generator
        std::random_device rd;
        generator = std::mt19937(rd());

        initial_opinion_dist = std::uniform_int_distribution(static_cast<int>(RobotOpinion::white),
                                                             static_cast<int>(RobotOpinion::black));

        exploration_duration_dist = std::exponential_distribution(1.0 / exp_mean_dur_ticks);

        dissemination_duration_dist = std::exponential_distribution(1.0 / dis_mean_dur_fac_ticks); // dummy initialization, params will be overwritten later
    }

    /**
     * @brief Convert robot data into data string
     *
     * @param st Current robot state
     * @param q_white Quality of white opinion
     * @param q_black Quality of black opinion
     * @param op Current opinion
     * @return std::string Converted data string
     */
    inline std::string ConvertStateQualitiesOpinionToString(const int &st,
                                                            const double &q_white,
                                                            const double &q_black,
                                                            const int &op)
    {
        std::stringstream ss;
        ss.precision(6);

        ss << std::fixed << st << "," << q_white << "," << q_black << "," << op;

        return ss.str();
    }

    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    int num_robots;

    double exp_mean_dur_ticks; ///< exploration mean duration in units of ticks

    double dis_mean_dur_fac_ticks; ///< dissemination mean duration factor in units of ticks

    double spd;

    double b_prob;

    double w_prob;

    bool voter_model;

    bool initialized = false;

    RobotDataStruct current_robot_data;

    std::mt19937 generator; ///< Random number generator

    std::shared_ptr<RobotIdDataStrMap> id_data_str_map_ptr; ///< Pointer to the map connecting robot IDs and their vector of data strings

    std::set<std::string> initialized_robot_ids; ///< Set of robot IDs that have been initialized

    std::uniform_int_distribution<int> initial_opinion_dist; ///< D istribution for generating initial robot opinions

    std::exponential_distribution<double> exploration_duration_dist; ///< Distribution to sample the exploration duration

    std::exponential_distribution<double> dissemination_duration_dist; ///< Distribution to sample the dissemination duration

    std::exponential_distribution<double>::param_type dis_mean_rate_ticks; ///< Parameter used to scale the dissemination duration
};

struct BenchmarkDataValentini2016 : BenchmarkDataBase
{
    BenchmarkDataValentini2016() : BenchmarkDataBase(VALENTINI_2016) {}

    std::vector<double> &sensor_probability_range = benchmark_param_range;

    double exp_mean_duration; ///< exploration mean duration factor in units of seconds

    double dis_mean_duration_factor; ///< dissemination mean duration factor in units of seconds

    bool voter_model;
};

class BenchmarkValentini2016 : public BenchmarkAlgorithmTemplate<BenchmarkDataValentini2016>
{
public:
    BenchmarkValentini2016(const BuzzForeachVMFunc &buzz_foreach_vm_func,
                           TConfigurationNode &t_tree,
                           const std::vector<std::string> &robot_id_vec);

    inline void Init() {}

    void SetupExperiment(const int &trial_ind, const std::pair<double, double> &curr_paired_parameters);

    void PostStep();

    void PostExperiment(const bool &final_experiment);

    void InitializeJson();

    void SaveData(const std::string &foldername_prefix = "");

    inline std::vector<double> GetParameterRange() { return data_.sensor_probability_range; }

    inline std::string GetParameterKeyword() { return VALENTINI_2016_PARAM; }

    inline std::string GetParameterKeywordAbbr() { return VALENTINI_2016_PARAM_ABBR; }

private:
    ProcessRobotOpinions process_robot_opinions_functor_;

    std::shared_ptr<RobotIdDataStrMap> id_data_str_map_ptr_;
};

#endif