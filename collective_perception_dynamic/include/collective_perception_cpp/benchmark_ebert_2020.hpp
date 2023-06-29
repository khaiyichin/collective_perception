#ifndef BENCHMARK_EBERT_2020_HPP
#define BENCHMARK_EBERT_2020_HPP

#include "benchmark_algorithm.hpp"
#include "custom_beta_cdf_gsl.hpp"

// Define default benchmark algorithm identifiers; if modified then must change at other locations
#define EBERT_2020 std::string("ebert_2020")
#define EBERT_2020_PARAM std::string("sensor_probability")
#define EBERT_2020_PARAM_ABBR std::string("sp")

using RobotIdDataStrMap = std::unordered_map<std::string, std::vector<std::string>>; // at each time step, the data string would be "<ALPHA>,<BETA>,<POSTERIOR>,<DECISION>"

struct ProcessRobotPosteriorProb : public CBuzzLoopFunctions::COperation
{
    enum class RobotDecision
    {
        undecided = -1,
        white = 0,
        black = 1
    };

    ProcessRobotPosteriorProb() {}

    inline ProcessRobotPosteriorProb(const int &num_robots,
                                     const float &speed,
                                     const double &b_prob,
                                     const double &w_prob,
                                     const int &prior_param,
                                     const double &credible_threshold,
                                     const bool &positive_feedback,
                                     const std::shared_ptr<RobotIdDataStrMap> &id_data_str_map_ptr)
        : num_robots(num_robots),
          spd(speed),
          b_prob(b_prob),
          w_prob(w_prob),
          prior_param(prior_param),
          credible_threshold(credible_threshold),
          positive_feedback(positive_feedback),
          id_data_str_map_ptr(id_data_str_map_ptr)
    {
        // Initialize generator
        std::random_device rd;
        generator = std::mt19937(rd());
    }

    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    /**
     * @brief Evaluate the Beta CDF value
     * This is used to find the posterior probability P(0 < X < 0.5), as described in the paper
     *
     * @return double
     */
    inline double EvaluateBetaCDF(const double &beta_a, const double &beta_b, const double &x = 0.5)
    {
        return CustomBetaCdfGSL::gsl_cdf_beta_P(x, beta_a, beta_b);
    }

    inline std::string ConvertParamPosteriorDecisionToString(const int &alpha,
                                                             const int &beta,
                                                             const double &posterior_prob,
                                                             const int &decision_int)
    {
        std::stringstream ss;
        ss.precision(6);

        ss << std::fixed << alpha << "," << beta << "," << posterior_prob << "," << decision_int;

        return ss.str();
    }

    int num_robots;

    int num_decided_robots = 0;

    int prior_param;

    double spd;

    double b_prob;

    double w_prob;

    double credible_threshold;

    bool positive_feedback;

    bool initialized = false;

    bool decided = false;

    std::mt19937 generator; ///< Random number generator

    std::shared_ptr<RobotIdDataStrMap> id_data_str_map_ptr;

    std::set<std::string> initialized_robot_ids; ///< Set of robot IDs that have been initialized
};

struct BenchmarkDataEbert2020 : BenchmarkDataBase
{
    BenchmarkDataEbert2020() : BenchmarkDataBase(EBERT_2020) {}

    std::vector<double> &sensor_probability_range = benchmark_param_range;

    int prior_param;

    double credible_threshold;

    bool positive_feedback;
};

class BenchmarkEbert2020 : public BenchmarkAlgorithmTemplate<BenchmarkDataEbert2020>
{
public:
    BenchmarkEbert2020(const BuzzForeachVMFunc &buzz_foreach_vm_func,
                       TConfigurationNode &t_tree,
                       const std::vector<std::string> &robot_id_vec);

    void Init() {}

    void SetupExperiment(const int &trial_ind, const std::pair<double, double> &curr_paired_parameters);

    void PostStep();

    void PostExperiment(const bool &final_experiment);

    void InitializeJson();

    void SaveData(const std::string &foldername_prefix = "");

    inline std::vector<double> GetParameterRange() { return data_.sensor_probability_range; }

    inline std::string GetParameterKeyword() { return EBERT_2020_PARAM; }

    inline std::string GetParameterKeywordAbbr() { return EBERT_2020_PARAM_ABBR; }

private:
    ProcessRobotPosteriorProb process_robot_posterior_prob_functor_;

    std::shared_ptr<RobotIdDataStrMap> id_data_str_map_ptr_;
};

#endif