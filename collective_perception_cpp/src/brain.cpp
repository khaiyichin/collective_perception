#include "brain.hpp"
#include <iostream>

void Brain::Solver::LocalSolve(const int &total_b_obs, const int &total_obs, const float &b, const float &w)
{
    float h = static_cast<float>(total_b_obs);
    float t = static_cast<float>(total_obs);

    if ((b == 1.0) && (w == 1.0)) // perfect sensor
    {
        local_vals.x = (float)h / t;
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
            local_vals.x = ((float)h / t + w - 1.0) / (b + w - 1);
            local_vals.confidence = (std::pow(t, 3) * std::pow(b + w - 1.0, 2)) / (h * (t - h));
        }
    }
}

void Brain::Solver::SocialSolve(const std::vector<ValuePair> &neighbor_vals)
{
    // Accumulate all the values in a sum
    auto lambda = [](const ValuePair &a, const ValuePair &b)
    {
        return Brain::ValuePair(a.x + b.x, (a.confidence * b.confidence) / (a.confidence + b.confidence));
    };

    Brain::ValuePair sum = std::reduce(neighbor_vals.begin(), neighbor_vals.end(), Brain::ValuePair(), lambda);

    // Assign the averages as social values
    social_vals.x = sum.x / neighbor_vals.size();
    social_vals.confidence = sum.confidence * neighbor_vals.size();
}

void Brain::Solver::InformedSolve()
{
    informed_vals.x = (local_vals.confidence * local_vals.x + social_vals.confidence * social_vals.x) / (local_vals.confidence + social_vals.confidence);
    informed_vals.confidence = local_vals.confidence + social_vals.confidence;
}

void Brain::Solve()
{
    solver_.LocalSolve(total_black_obs_, total_obs_, b_acc_, w_acc_);
    solver_.SocialSolve(neighbors_value_pairs_);
    solver_.InformedSolve();
}