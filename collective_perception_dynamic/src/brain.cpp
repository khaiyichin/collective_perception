#include "brain.hpp"

void Brain::Solver::LocalSolve(const int &total_b_obs, const int &total_obs, const float &b, const float &w)
{
    float h = static_cast<float>(total_b_obs);
    float t = static_cast<float>(total_obs);

    if ((b == 1.0) && (w == 1.0)) // perfect sensor
    {
        local_vals.x = h / t;
        local_vals.confidence = std::pow(t, 3) / (h * (t - h));
    }
    else // imperfect sensor
    {
        if (h <= (1.0 - w) * t)
        {
            float num = std::pow(b + w - 1.0, 2) * (t * std::pow(w, 2) - 2 * (t - h) * w + (t - h));
            float denom = std::pow(w, 2) * std::pow(w - 1.0, 2);

            local_vals.x = 0.0;
            local_vals.confidence = num / denom;
        }
        else if (h >= b * t)
        {
            float num = std::pow(b + w - 1.0, 2) * (t * std::pow(b, 2) - 2 * h * b + h);
            float denom = std::pow(b, 2) * std::pow(b - 1.0, 2);

            local_vals.x = 1.0;
            local_vals.confidence = num / denom;
        }
        else
        {
            local_vals.x = (h / t + w - 1.0) / (b + w - 1);
            local_vals.confidence = (std::pow(t, 3) * std::pow(b + w - 1.0, 2)) / (h * (t - h));
        }
    }
}

void Brain::Solver::SocialSolve(const std::vector<ValuePair> &neighbor_vals, const bool &legacy)
{
    Brain::ValuePair sum;

    // Check whether to compute using legacy equations
    if (legacy)
    {
        // Accumulate all the values in a sum (the confidence is the inverse of variance)
        auto lambda = [](const ValuePair &a, const ValuePair &b) -> Brain::ValuePair
        {
            if (a.confidence == 0.0 && b.confidence == 0.0)
            {
                return Brain::ValuePair(0.0, 0.0);
            }
            else if (a.confidence == 0.0)
            {
                return b;
            }
            else if (b.confidence == 0.0)
            {
                return a;
            }
            else
            {
                return Brain::ValuePair(a.x + b.x, (a.confidence * b.confidence) / (a.confidence + b.confidence));
            }
        };

        sum = std::reduce(neighbor_vals.begin(), neighbor_vals.end(), Brain::ValuePair(), lambda);

        // Assign the averages as social values
        social_vals.x = sum.x / neighbor_vals.size();
        social_vals.confidence = sum.confidence * neighbor_vals.size();
    }
    else
    {
        auto lambda = [](const ValuePair &left, const ValuePair &right) -> Brain::ValuePair
        {
            return Brain::ValuePair(left.x + right.x * right.confidence, left.confidence + right.confidence);
        };

        sum = std::accumulate(neighbor_vals.begin(), neighbor_vals.end(), Brain::ValuePair(0.0, 0.0), lambda);

        // Assign the averages as social values
        social_vals.x = (sum.confidence == 0.0) ? 0.0 : sum.x / sum.confidence;
        social_vals.confidence = sum.confidence;
    }
}

void Brain::Solver::InformedSolve()
{
    if (local_vals.confidence == 0.0 && social_vals.confidence == 0.0)
    {
        informed_vals.x = local_vals.x;
    }
    else
    {
        informed_vals.x = (local_vals.confidence * local_vals.x + social_vals.confidence * social_vals.x) / (local_vals.confidence + social_vals.confidence);
    }

    informed_vals.confidence = local_vals.confidence + social_vals.confidence;
}

void Brain::Solve()
{
    // Solve local values (since it's purely based on self observations)
    solver_.LocalSolve(total_black_obs_, total_obs_, b_prob_, w_prob_);

    // Compute social values only if neighbor values are available
    if (neighbors_value_pairs_.size() != 0)
    {
        solver_.SocialSolve(neighbors_value_pairs_, legacy_);
    }

    // Solve informed values (since local values are always available, even if social values aren't)
    solver_.InformedSolve();
}

void Brain::Disable()
{
    // Apply dummy values
    StoreObservations(-1, -1);

    solver_.local_vals.x = -1.0;
    solver_.local_vals.confidence = -1.0;
    solver_.social_vals.x = -1.0;
    solver_.social_vals.confidence = -1.0;
    solver_.informed_vals.x = -1.0;
    solver_.informed_vals.confidence = -1.0;

    // Clear the log of neighboring values
    StoreNeighborValuePairs(std::vector<Brain::ValuePair>{});

    // Set disabled status
    disabled_ = true;
}