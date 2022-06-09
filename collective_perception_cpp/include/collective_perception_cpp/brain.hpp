#ifndef BRAIN_HPP
#define BRAIN_HPP

#include <string>
#include <vector>
#include <cmath>
#include <numeric>

/**
 * @brief Class to perform robot computation
 *
 */
class Brain
{
public:
    /**
     * @brief Struct to store estimate and confidence pair
     *
     */
    struct ValuePair
    {
        /**
         * @brief Construct a new ValuePair struct
         *
         */
        ValuePair() : x(0.0), confidence(1.0) {}

        /**
         * @brief Construct a new ValuePair struct
         *
         * @param x
         * @param confidence
         */
        ValuePair(const float &x, const float &confidence) : x(x), confidence(confidence) {}

        float x; ///< Estimate value

        float confidence; ///< Confidence value
    };

    ///< @todo: Create an abstract Solver class to allow for various functions to be used

    /**
     * @brief Struct to store and solve estimates and confidences
     *
     */
    struct Solver
    {
        /**
         * @brief Construct a new Solver object
         *
         */
        Solver() {}

        /**
         * @brief Solve for local values
         *
         * @param total_b_obs Total number of black tiles observed
         * @param total_obs Total number observations made
         * @param b Sensor accuracy for black tiles
         * @param w Sensor accuracy for white tiles
         */
        void LocalSolve(const int &total_b_obs, const int &total_obs, const float &b, const float &w);

        /**
         * @brief Solve for social values
         *
         * @param neighbor_vals Vector of neighbor values
         */
        void SocialSolve(const std::vector<ValuePair> &neighbor_vals);

        /**
         * @brief Solve for informed values
         *
         */
        void InformedSolve();

        ValuePair local_vals; ///< Local values

        ValuePair social_vals; ///< Social values

        ValuePair informed_vals; ///< Informed values
    };

    /**
     * @brief Construct a new Brain object
     *
     */
    Brain(){};

    /**
     * @brief Construct a new Brain object
     *
     * @param id ID of the robot
     * @param b_acc Sensor accuracy for black tiles
     * @param w_acc Sensor accuracy for white tiles
     */
    Brain(const std::string &id, const float &b_acc, const float &w_acc) : id_(id), b_acc_(b_acc), w_acc_(w_acc){};

    /**
     * @brief Get the local ValuePair object
     *
     * @return ValuePair The local values
     */
    ValuePair GetLocalValuePair() { return solver_.local_vals; }

    /**
     * @brief Get the social ValuePair object
     *
     * @return ValuePair The social values
     */
    ValuePair GetSocialValuePair() { return solver_.social_vals; }

    /**
     * @brief Get the informed ValuePair object
     *
     * @return ValuePair The informed values
     */
    ValuePair GetInformedValuePair() { return solver_.informed_vals; }

    /**
     * @brief Get the robot's ID that's associated with the Brain instance
     *
     * @return std::string Robot ID
     */
    std::string GetId() { return id_; }

    /**
     * @brief Solve values
     *
     */
    void Solve();

    /**
     * @brief Store robot observations
     *
     * @param total_b_obs Total number of black tiles observed
     * @param total_obs Total number of observations made
     */
    void StoreObservations(const int &total_b_obs, const int &total_obs)
    {
        total_black_obs_ = total_b_obs;
        total_obs_ = total_obs;
    }

    /**
     * @brief Store the local values
     *
     * @param value_pair Local ValuePair object
     */
    void StoreLocalValuePair(const ValuePair &value_pair) { solver_.local_vals = value_pair; }

    /**
     * @brief Store the neighbors' values
     *
     * @param value_pair_vec Vector of neighbor's ValuePair objects
     */
    void StoreNeighborValuePairs(const std::vector<ValuePair> &value_pair_vec) { neighbors_value_pairs_ = value_pair_vec; }

private:
    std::string id_; ///< Robot ID

    int total_black_obs_; ///< Total number of black tiles observed

    int total_obs_; ///< Total observations made

    float b_acc_; ///< Sensor accuracy for observing black tiles

    float w_acc_; ///< Sensor accuracy for observing white tiles

    std::vector<ValuePair> neighbors_value_pairs_; ///< Vector of neighbors' values

    Solver solver_; ///< Solver object
};

#endif