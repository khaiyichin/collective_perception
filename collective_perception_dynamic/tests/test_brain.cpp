#include "test_header.hpp"
#include "brain.hpp"

TEST_CASE("Brain class", "[BrainCls]")
{
    // Initialize random number generator
    // std::srand(std::time(nullptr));
    std::random_device rd;
    std::default_random_engine generator = std::default_random_engine(rd());
    std::uniform_real_distribution<float> sensor_distribution(0.0, 1.0);
    std::uniform_real_distribution<float> confidence_distribution(0, CONFIDENCE_MAX);
    std::uniform_int_distribution<int> neighbor_distribution(1, NEIGHBOR_AGENTS_MAX);
    std::uniform_int_distribution<int> observation_distribution(0, OBSERVATION_MAX);

    SECTION("test brain class initialization", "[initialization]")
    {
        Brain b;

        Brain::ValuePair local_vals = b.GetLocalValuePair();
        Brain::ValuePair social_vals = b.GetSocialValuePair();
        Brain::ValuePair informed_vals = b.GetInformedValuePair();

        REQUIRE(local_vals.x == 0.5);
        REQUIRE(local_vals.confidence == 0.0);
        REQUIRE(social_vals.x == 0.0);
        REQUIRE(social_vals.confidence == 0.0);
        REQUIRE(informed_vals.x == 0.0);
        REQUIRE(informed_vals.confidence == 0.0);
    }

    SECTION("test agent estimate and confidence computation", "[compute_values]")
    {
        // Generate random sensor accuracy/probability
        float b = sensor_distribution(generator);
        float w = b;

        // Create Brain instance
        Brain brain("test", b, w, false);

        std::vector<float> manual_local_est, manual_local_conf, manual_social_est, manual_social_conf, manual_informed_est, manual_informed_conf;
        std::vector<float> class_local_est, class_local_conf, class_social_est, class_social_conf, class_informed_est, class_informed_conf;

        // Repeat computation multiple times to get a larger sample
        for (size_t i = 0; i < TEST_REPEATS; ++i)
        {
            // Generate random number of observations
            float t = observation_distribution(generator);                                   // total tiles observed
            float n = std::round(observation_distribution(generator) / OBSERVATION_MAX * t); // total black tiles observed
            float local_est, local_conf;

            // Compute local values manually
            if (n <= (1.0 - w) * t)
            {
                local_est = 0.0;

                float num_conf = std::pow(b + w - 1.0, 2) * (t * std::pow(w, 2) - 2 * (t - n) * w + t - n);
                float denom_conf = std::pow(w, 2) * std::pow(w - 1.0, 2);
                local_conf = num_conf / denom_conf;
            }
            else if (n >= b * t)
            {
                local_est = 1.0;

                float num_conf = std::pow(b + w - 1.0, 2) * (t * std::pow(b, 2) - 2 * n * b + n);
                float denom_conf = std::pow(b, 2) * std::pow(b - 1.0, 2);
                local_conf = num_conf / denom_conf;
            }
            else
            {
                local_est = (n / t + w - 1.0) / (b + w - 1.0);

                float num_conf = std::pow(b + w - 1.0, 2) * std::pow(t, 3);
                float denom_conf = n * (t - n);
                local_conf = num_conf / denom_conf;
            }

            manual_local_est.push_back(local_est);
            manual_local_conf.push_back(local_conf);

            // Generate random number of neighbor values
            if (sensor_distribution(generator) >= 0.5) // randomize communication occurrence with 0.5 probability
            {
                int neighbors = neighbor_distribution(generator); // number of neighbors
                std::vector<Brain::ValuePair> neighbor_vals;

                for (size_t j = 0; j < neighbors; ++j)
                {
                    neighbor_vals.push_back(
                        Brain::ValuePair(sensor_distribution(generator), // equivalent to fill ratio anyway
                                         confidence_distribution(generator)));
                }

                // Store neighbor values
                brain.StoreNeighborValuePairs(neighbor_vals);

                // Compute social values manually
                float social_est = 0.0, social_conf = 0.0;

                for (auto &v : neighbor_vals)
                {
                    social_est += v.x * v.confidence;
                    social_conf += v.confidence;
                }

                manual_social_est.push_back((social_conf == 0.0) ? 0.0 : social_est / social_conf);
                manual_social_conf.push_back(social_conf);
            }
            else
            {
                manual_social_est.push_back((manual_social_est.size() > 0) ? manual_social_est.back() : 0.0);
                manual_social_conf.push_back((manual_social_conf.size() > 0) ? manual_social_conf.back() : 0.0);
            }

            // Compute informed values manually
            manual_informed_est.push_back((local_est * local_conf + manual_social_est[i] * manual_social_conf[i]) /
                                          (local_conf + manual_social_conf[i]));
            manual_informed_conf.push_back((local_conf + manual_social_conf[i]) / 2);

            // Compute local, social and informed values using Brain class
            brain.StoreObservations(n, t);
            brain.Solve();

            // Extract and store Brain class values
            Brain::ValuePair local_vals = brain.GetLocalValuePair();
            Brain::ValuePair social_vals = brain.GetSocialValuePair();
            Brain::ValuePair informed_vals = brain.GetInformedValuePair();

            class_local_est.push_back(local_vals.x);
            class_local_conf.push_back(local_vals.confidence);
            class_social_est.push_back(social_vals.x);
            class_social_conf.push_back(social_vals.confidence);
            class_informed_est.push_back(informed_vals.x);
            class_informed_conf.push_back(informed_vals.confidence);
        }

        // Check local values
        REQUIRE(std::all_of(manual_local_est.begin(), manual_local_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE(std::all_of(class_local_est.begin(), class_local_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE_THAT(class_local_est, Catch::Matchers::Approx(manual_local_est).margin(1e-3));

        REQUIRE(std::all_of(manual_local_conf.begin(), manual_local_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE(std::all_of(class_local_conf.begin(), class_local_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE_THAT(class_local_conf, Catch::Matchers::Approx(manual_local_conf).margin(1e-1));

        // Check social values
        REQUIRE(std::all_of(manual_social_est.begin(), manual_social_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE(std::all_of(class_social_est.begin(), class_social_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE_THAT(class_social_est, Catch::Matchers::Approx(manual_social_est).margin(1e-3));

        REQUIRE(std::all_of(manual_social_conf.begin(), manual_social_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE(std::all_of(class_social_conf.begin(), class_social_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE_THAT(class_social_conf, Catch::Matchers::Approx(manual_social_conf).margin(1e-1));

        // Check informed values
        REQUIRE(std::all_of(manual_informed_est.begin(), manual_informed_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE(std::all_of(class_informed_est.begin(), class_informed_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE_THAT(class_informed_est, Catch::Matchers::Approx(manual_informed_est).margin(1e-3));

        REQUIRE(std::all_of(manual_informed_conf.begin(), manual_informed_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE(std::all_of(class_informed_conf.begin(), class_informed_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE_THAT(class_informed_conf, Catch::Matchers::Approx(manual_informed_conf).margin(1e-1));
    }

    SECTION("test legacy agent estimate and confidence computation", "[compute_values]")
    {
        // Generate random sensor accuracy/probability
        float b = sensor_distribution(generator);
        float w = b;

        // Create Brain instance
        Brain brain("test", b, w, true);

        std::vector<float> manual_local_est, manual_local_conf, manual_social_est, manual_social_conf, manual_informed_est, manual_informed_conf;
        std::vector<float> class_local_est, class_local_conf, class_social_est, class_social_conf, class_informed_est, class_informed_conf;

        // Repeat computation multiple times to get a larger sample
        for (size_t i = 0; i < TEST_REPEATS; ++i)
        {
            // Generate random number of observations
            float t = observation_distribution(generator);                                   // total tiles observed
            float n = std::round(observation_distribution(generator) / OBSERVATION_MAX * t); // total black tiles observed
            float local_est, local_conf;

            // Compute local values manually
            if (n <= (1.0 - w) * t)
            {
                local_est = 0.0;

                float num_conf = std::pow(b + w - 1.0, 2) * (t * std::pow(w, 2) - 2 * (t - n) * w + t - n);
                float denom_conf = std::pow(w, 2) * std::pow(w - 1.0, 2);
                local_conf = num_conf / denom_conf;
            }
            else if (n >= b * t)
            {
                local_est = 1.0;

                float num_conf = std::pow(b + w - 1.0, 2) * (t * std::pow(b, 2) - 2 * n * b + n);
                float denom_conf = std::pow(b, 2) * std::pow(b - 1.0, 2);
                local_conf = num_conf / denom_conf;
            }
            else
            {
                local_est = (n / t + w - 1.0) / (b + w - 1.0);

                float num_conf = std::pow(b + w - 1.0, 2) * std::pow(t, 3);
                float denom_conf = n * (t - n);
                local_conf = num_conf / denom_conf;
            }

            manual_local_est.push_back(local_est);
            manual_local_conf.push_back(local_conf);

            // Generate random number of neighbor values
            if (sensor_distribution(generator) >= 0.5) // randomize communication occurrence with 0.5 probability
            {
                int neighbors = neighbor_distribution(generator); // number of neighbors
                std::vector<Brain::ValuePair> neighbor_vals;

                for (size_t j = 0; j < neighbors; ++j)
                {
                    neighbor_vals.push_back(
                        Brain::ValuePair(sensor_distribution(generator), // equivalent to fill ratio anyway
                                         confidence_distribution(generator)));
                }

                // Store neighbor values
                brain.StoreNeighborValuePairs(neighbor_vals);

                // Compute social values manually
                float social_est = 0.0, social_conf = 0.0;

                for (auto &v : neighbor_vals)
                {
                    social_est += v.x;
                    social_conf += (v.confidence == 0.0) ? 0.0 : 1 / v.confidence;
                }

                manual_social_est.push_back(social_est / neighbors);
                manual_social_conf.push_back((social_conf == 0.0) ? 0.0 : neighbors * 1.0 / social_conf);
            }
            else
            {
                manual_social_est.push_back((manual_social_est.size() > 0) ? manual_social_est.back() : 0.0);
                manual_social_conf.push_back((manual_social_conf.size() > 0) ? manual_social_conf.back() : 0.0);
            }

            // Compute informed values manually
            manual_informed_est.push_back((local_est * local_conf + manual_social_est[i] * manual_social_conf[i]) /
                                          (local_conf + manual_social_conf[i]));
            manual_informed_conf.push_back(local_conf + manual_social_conf[i]);

            // Compute local, social and informed values using Brain class
            brain.StoreObservations(n, t);
            brain.Solve();

            // Extract and store Brain class values
            Brain::ValuePair local_vals = brain.GetLocalValuePair();
            Brain::ValuePair social_vals = brain.GetSocialValuePair();
            Brain::ValuePair informed_vals = brain.GetInformedValuePair();

            class_local_est.push_back(local_vals.x);
            class_local_conf.push_back(local_vals.confidence);
            class_social_est.push_back(social_vals.x);
            class_social_conf.push_back(social_vals.confidence);
            class_informed_est.push_back(informed_vals.x);
            class_informed_conf.push_back(informed_vals.confidence);
        }

        // Check local values
        REQUIRE(std::all_of(manual_local_est.begin(), manual_local_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE(std::all_of(class_local_est.begin(), class_local_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE_THAT(class_local_est, Catch::Matchers::Approx(manual_local_est).margin(1e-3));

        REQUIRE(std::all_of(manual_local_conf.begin(), manual_local_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE(std::all_of(class_local_conf.begin(), class_local_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE_THAT(class_local_conf, Catch::Matchers::Approx(manual_local_conf).margin(1e-1));

        // Check social values
        REQUIRE(std::all_of(manual_social_est.begin(), manual_social_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE(std::all_of(class_social_est.begin(), class_social_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE_THAT(class_social_est, Catch::Matchers::Approx(manual_social_est).margin(1e-3));

        REQUIRE(std::all_of(manual_social_conf.begin(), manual_social_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE(std::all_of(class_social_conf.begin(), class_social_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE_THAT(class_social_conf, Catch::Matchers::Approx(manual_social_conf).margin(1e-1));

        // Check informed values
        REQUIRE(std::all_of(manual_informed_est.begin(), manual_informed_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE(std::all_of(class_informed_est.begin(), class_informed_est.end(), [](const float &value)
                            { return (value <= 1.0 && value >= 0.0); }));
        REQUIRE_THAT(class_informed_est, Catch::Matchers::Approx(manual_informed_est).margin(1e-3));

        REQUIRE(std::all_of(manual_informed_conf.begin(), manual_informed_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE(std::all_of(class_informed_conf.begin(), class_informed_conf.end(), [](const float &a)
                            { return a >= 0.0; }));
        REQUIRE_THAT(class_informed_conf, Catch::Matchers::Approx(manual_informed_conf).margin(1e-1));
    }
}