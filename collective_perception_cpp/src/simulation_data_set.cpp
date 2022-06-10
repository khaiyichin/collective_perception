#include "simulation_data_set.hpp"

SimulationDataSet::SimulationDataSet(collective_perception_cpp::proto::SimulationDataSet &sim_data_set_msg)
{
    // Store simulation parameters
    num_agents_ = sim_data_set_msg.num_agents();   // number of agents
    num_experiments_ = sim_data_set_msg.num_exp(); // number of experiments
    num_steps_ = sim_data_set_msg.num_steps();     // number of simulation steps
    comms_range_ = sim_data_set_msg.comms_range(); // communication range
    density_ = sim_data_set_msg.density();         // swarm density

    std::copy(sim_data_set_msg.tfr_range().begin(),
              sim_data_set_msg.tfr_range().end(),
              std::back_inserter(tfr_range_)); // range of target fill ratios
    std::copy(sim_data_set_msg.sp_range().begin(),
              sim_data_set_msg.sp_range().end(),
              std::back_inserter(sp_range_)); // range of sensor probabilities

    // Store SimPacket objects
    for (auto itr = sim_data_set_msg.mutable_sim_packets()->begin();
         itr != sim_data_set_msg.mutable_sim_packets()->end();
         ++itr)
    {
        SimPacket packet;

        packet.comms_range = itr->comms_range();
        packet.target_fill_ratio = itr->tfr();
        packet.b_prob = itr->b_prob();
        packet.w_prob = itr->w_prob();
        packet.num_agents = itr->num_agents();
        packet.num_experiments = itr->num_exp();
        packet.num_steps = itr->num_steps();
        packet.density = itr->density();

        // Store AgentData objects
        auto repeated_exp_agt_data_msg_ptr = itr->mutable_agent_data(); // pointer to collective_perception_cpp::proto::RepeatedExperimentAgentData type

        for (auto itr_agt = repeated_exp_agt_data_msg_ptr->mutable_agents()->begin();
             itr_agt != repeated_exp_agt_data_msg_ptr->mutable_agents()->end();
             ++itr_agt)
        {
            AgentData a;

            // Store vector of tile occurrences
            std::copy(itr_agt->tile_occurrences().begin(),
                      itr_agt->tile_occurrences().end(),
                      std::back_insert_iterator(a.tile_occurrences));

            // Store vector of observations
            std::copy(itr_agt->observations().begin(),
                      itr_agt->observations().end(),
                      std::back_insert_iterator(a.observations));

            packet.agent_data_vec.push_back(a);
        }

        // Store Stats objects (local, social, and informed)
        auto repeated_exp_stats_msg_ptr = itr->mutable_stats(); // pointer to collective_perception_cpp::proto::RepeatedExperimentStats type
        auto &local_vals_msg = repeated_exp_stats_msg_ptr->local_vals();
        auto &social_vals_msg = repeated_exp_stats_msg_ptr->social_vals();
        auto &informed_vals_msg = repeated_exp_stats_msg_ptr->informed_vals();

        for (size_t i = 0; i < num_steps_; ++i)
        {
            Stats local, social, informed;

            // Copy local values
            std::copy(local_vals_msg.Get(i).x_mean().begin(),
                      local_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(local.x_sample_mean));
            std::copy(local_vals_msg.Get(i).conf_mean().begin(),
                      local_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(local.confidence_sample_mean));
            std::copy(local_vals_msg.Get(i).x_std().begin(),
                      local_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(local.x_sample_std));
            std::copy(local_vals_msg.Get(i).conf_std().begin(),
                      local_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(local.confidence_sample_std));

            packet.local_values_vec.push_back(local);

            // Copy social values
            std::copy(social_vals_msg.Get(i).x_mean().begin(),
                      social_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(social.x_sample_mean));
            std::copy(social_vals_msg.Get(i).conf_mean().begin(),
                      social_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(social.confidence_sample_mean));
            std::copy(social_vals_msg.Get(i).x_std().begin(),
                      social_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(social.x_sample_std));
            std::copy(social_vals_msg.Get(i).conf_std().begin(),
                      social_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(social.confidence_sample_std));

            packet.social_values_vec.push_back(social);

            // Copy informed values
            std::copy(informed_vals_msg.Get(i).x_mean().begin(),
                      informed_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(informed.x_sample_mean));
            std::copy(informed_vals_msg.Get(i).conf_mean().begin(),
                      informed_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(informed.confidence_sample_mean));
            std::copy(informed_vals_msg.Get(i).x_std().begin(),
                      informed_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(informed.x_sample_std));
            std::copy(informed_vals_msg.Get(i).conf_std().begin(),
                      informed_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(informed.confidence_sample_std));

            packet.informed_values_vec.push_back(informed);
        }

        // Insert populated SimPacket object
        InsertSimPacket(packet);
    }
}

void SimulationDataSet::Serialize(collective_perception_cpp::proto::SimulationDataSet &sim_data_set_msg)
{
    sim_data_set_msg.set_num_agents(num_agents_);   // number of agents
    sim_data_set_msg.set_num_exp(num_experiments_); // number of experiments
    sim_data_set_msg.set_num_steps(num_steps_);     // number of simulation steps
    sim_data_set_msg.set_comms_range(comms_range_); // communication range
    sim_data_set_msg.set_density(density_);         // swarm density

    *sim_data_set_msg.mutable_tfr_range() = {tfr_range_.begin(), tfr_range_.end()};
    *sim_data_set_msg.mutable_sp_range() = {sp_range_.begin(), sp_range_.end()};

    std::vector<SimPacket> sim_packets_vec = GetAllSimPackets();
}

SimPacket SimulationDataSet::GetSimPacket(const float &tfr, const float &sp)
{
    /* Internal units description

    The values for tfr and sp range from 0.000 to 1.000, with a resolution of 0.005.
    Therefore, after scaling to integers, the resolution for the internal units would be 5, and the new
    range is between 0 to 1000.
    */

    // Convert to internal units to use as keys (thus avoiding trailing values with floats)
    unsigned int tfr_internal, sp_internal;

    tfr_internal = RoundToMultiple(ConvertToInternalUnits(tfr), 5);
    sp_internal = RoundToMultiple(ConvertToInternalUnits(sp), 5);

    return sim_packets_[tfr_internal][sp_internal];
}

void SimulationDataSet::InsertSimPacket(const SimPacket &packet)
{
    int tfr_internal = RoundToMultiple(ConvertToInternalUnits(packet.target_fill_ratio), 5);
    int sp_internal = RoundToMultiple(ConvertToInternalUnits(packet.b_prob), 5);

    sim_packets_[tfr_internal][sp_internal] = packet;
}

std::vector<SimPacket> SimulationDataSet::GetAllSimPackets()
{
    std::vector<SimPacket> packets;

    for (auto &tfr_key_value : sim_packets_)
    {
        for (auto &sp_key_value : tfr_key_value.second)
        {
            packets.push_back(sp_key_value.second);
        }
    }

    return packets;
}

int SimulationDataSet::RoundToMultiple(const int &value, const int &base)
{
    int absBase = std::abs(base);
    int remainder = std::abs(value) % absBase;

    if (2 * remainder >= absBase)
    {
        return std::round(std::copysign(std::abs(value) + absBase - remainder, value));
    }
    else
    {
        return std::copysign(std::abs(value) - remainder, value);
    }
}