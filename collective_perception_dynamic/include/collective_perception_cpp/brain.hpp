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
        ValuePair() : x(0.0), confidence(0.0) {}

        /**
         * @brief Construct a new ValuePair struct
         *
         * @param x
         * @param confidence
         */
        ValuePair(const float &x, const float &confidence) : x(x), confidence(confidence) {}

        inline ValuePair operator/(const float &val)
        {
            return ValuePair(x / val, confidence / val);
        }

        friend inline bool operator==(const ValuePair &lhs, const ValuePair &rhs)
        {
            if (lhs.x == rhs.x && lhs.confidence == rhs.confidence)
            {
                return true;
            }

            else
            {
                return false;
            }
        }

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
        Solver() : local_vals(ValuePair(0.5, 0.0)) {}

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
         * @param legacy_mode Flag to indicate if using legacy equations
         */
        void SocialSolve(const std::vector<ValuePair> &neighbor_vals, const bool &legacy_mode);

        /**
         * @brief Solve for informed values
         *
         */
        void InformedSolve(const bool &legacy);

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
     * @param b_prob Sensor accuracy for black tiles
     * @param w_prob Sensor accuracy for white tiles
     */
    Brain(const std::string &id, const float &b_prob, const float &w_prob, const bool &legacy = false)
        : id_(id), b_prob_(b_prob), w_prob_(w_prob), legacy_(legacy){};

    /**
     * @brief Get the local ValuePair object
     *
     * @return ValuePair The local values
     */
    inline ValuePair GetLocalValuePair() { return solver_.local_vals; }

    /**
     * @brief Get the social ValuePair object
     *
     * @return ValuePair The social values
     */
    inline ValuePair GetSocialValuePair() { return solver_.social_vals; }

    /**
     * @brief Get the informed ValuePair object
     *
     * @return ValuePair The informed values
     */
    inline ValuePair GetInformedValuePair() { return solver_.informed_vals; }

    /**
     * @brief Get the robot's ID that's associated with the Brain instance
     *
     * @return std::string Robot ID
     */
    inline std::string GetId() { return id_; }

    inline float GetBProb() { return b_prob_; }

    inline float GetWProb() { return w_prob_; }

    inline bool GetSolverMode() { return legacy_; }

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
     * @brief Store the neighbors' values
     *
     * @param value_pair_vec Vector of neighbor's ValuePair objects
     */
    void StoreNeighborValuePairs(const std::vector<ValuePair> &value_pair_vec) { neighbors_value_pairs_ = value_pair_vec; }

private:
    bool legacy_; ///< legacy solver mode @todo may be removed in the future

    std::string id_; ///< Robot ID

    int total_black_obs_; ///< Total number of black tiles observed

    int total_obs_; ///< Total observations made

    float b_prob_; ///< Sensor accuracy for observing black tiles

    float w_prob_; ///< Sensor accuracy for observing white tiles

    std::vector<ValuePair> neighbors_value_pairs_; ///< Vector of neighbors' values

    Solver solver_; ///< Solver object
};

#endif